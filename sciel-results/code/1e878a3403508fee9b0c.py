# EVOLVE-BLOCK-START
from fractions import Fraction
import heapq
import random

def construct():
    recips = [Fraction(1, i + 1) for i in range(30)]
    
    # R[k] = sum_{i=k+1}^{30} 1/i
    R = [Fraction(0, 1)] * 31
    for k in range(29, -1, -1):
        R[k] = R[k + 1] + recips[k]
        
    def calc_sum(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * recips[i]
        return s

    def local_search(coeff, max_iters=10):
        current_coeff = coeff[:]
        current_sum = calc_sum(current_coeff)
        if current_sum == 0:
            # Perturb
            for i in range(29, -1, -1):
                current_coeff[i] = 1 if current_coeff[i] != 1 else -1
                current_sum = calc_sum(current_coeff)
                if current_sum != 0: break
            if current_sum == 0:
                current_coeff[29] = 0
                current_sum = calc_sum(current_coeff)
                if current_sum == 0:
                    return None, None

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_move = None
            best_val = None
            
            # 1-flip
            for i in range(30):
                old_val = current_coeff[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val: continue
                    delta = (new_val - old_val) * recips[i]
                    new_sum = current_sum + delta
                    if new_sum == 0: continue
                    if best_val is None or abs(new_sum) < abs(best_val):
                        best_val = new_sum
                        best_move = (i, new_val, 0, 0) # 0,0 to distinguish from 2-flip? No, just store tuple
                        best_move = (i, new_val)
            
            if best_move is not None:
                i, new_val = best_move
                current_coeff[i] = new_val
                current_sum = best_val
                improved = True
                continue
                
            # 2-flip (optional, maybe skip for speed in init phase)
            # For init phase, 1-flip is usually enough to get close.
            
        if current_sum == 0:
            return None, None
        return current_coeff, abs(current_sum)

    # Phase 1: Get a very good initial upper bound
    best_abs = Fraction(1, 1)
    best_coeff = None
    
    # Greedy
    c_greedy = [0] * 30
    s_greedy = Fraction(0, 1)
    for i in range(29, -1, -1):
        best_c = 0
        best_s = s_greedy
        for c in [-1, 0, 1]:
            s_new = s_greedy + c * recips[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy[i] = best_c
        s_greedy = best_s
    if s_greedy != 0:
        res_c, res_s = local_search(c_greedy, max_iters=5)
        if res_s is not None and res_s < best_abs:
            best_abs = res_s
            best_coeff = res_c

    # Random restarts for better init
    random.seed(42)
    for _ in range(50):
        coeff = [0] * 30
        num_nonzero = random.randint(10, 20)
        indices = random.sample(range(30), num_nonzero)
        for idx in indices:
            coeff[idx] = random.choice([-1, 1])
        res_c, res_s = local_search(coeff, max_iters=10)
        if res_s is not None and res_s < best_abs:
            best_abs = res_s
            best_coeff = res_c

    # Phase 2: Branch and Bound to find global optimum
    # If best_abs is already very small, B&B will be fast.
    
    counter = 0
    heap = [(Fraction(0, 1), counter, Fraction(0, 1), 0, ())]
    
    # Limit nodes to prevent timeout if pruning is weak
    # But with good init, it should be fine.
    # Let's add a node limit of 10000 to be safe?
    # If we hit the limit, we return the best found so far.
    node_limit = 20000
    nodes_processed = 0
    
    while heap and nodes_processed < node_limit:
        lb, _, s, depth, coeff = heapq.heappop(heap)
        nodes_processed += 1
        
        if lb >= best_abs:
            break
            
        if depth == 30:
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs:
                    best_abs = abs_s
                    best_coeff = list(coeff)
            continue
            
        next_recip = recips[depth]
        next_R = R[depth + 1]
        
        for c in [-1, 0, 1]:
            s_new = s + c * next_recip
            if s_new - next_R <= 0 <= s_new + next_R:
                lb_new = Fraction(0, 1)
            else:
                lb_new = abs(s_new) - next_R
                if lb_new < 0: lb_new = Fraction(0, 1)
                
            if lb_new < best_abs:
                counter += 1
                heapq.heappush(heap, (lb_new, counter, s_new, depth + 1, coeff + (c,)))
                
    if best_coeff is None:
        best_coeff = [0] * 30
        best_coeff[0] = 1
        
    return best_coeff
# EVOLVE-BLOCK-END
