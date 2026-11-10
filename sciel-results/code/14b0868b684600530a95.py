"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    def is_comparable(a, b):
        if a == b: return 0
        return 1 if ((a & b) == a or (a & b) == b) else 0
    
    # Precompute comparability for all pairs using bit operations
    # comp[i][j] = 1 if i and j are comparable (one is strict subset of other)
    # We'll use a list of bitsets for efficiency
    
    # For each mask, precompute which other masks it's comparable to
    # Using bitset representation: comp_bits[i] is a 4096-bit integer where bit j is 1 if i and j are comparable
    comp_bits = [0] * total_masks
    
    for i in range(total_masks):
        bits = 0
        for j in range(i+1, total_masks):
            if is_comparable(i, j):
                bits |= (1 << j)
        comp_bits[i] = bits
    
    # Precompute degree (number of comparable elements) for each mask
    deg_pre = [0] * total_masks
    for i in range(total_masks):
        deg_pre[i] = bin(comp_bits[i]).count('1')
    
    def popcount(x):
        return bin(x).count('1')
    
    # Min-cut based Maximum Weight Closure
    # We want to select exactly m nodes to maximize sum of comparable edges among selected nodes.
    # This is equivalent to: maximize (sum of weights of selected nodes) subject to closure constraints.
    # 
    # Lagrangian approach: For a penalty λ per node, maximize:
    #   sum_{i in S} (deg_pre[i] - λ) + (1/2) * sum_{i,j in S, i!=j} is_comparable(i,j)
    # But this double-counts. Let's think differently.
    #
    # Actually, the objective is: maximize (1/2) * sum_{i,j in S} is_comparable(i,j) = number of comparable pairs.
    # 
    # For Maximum Weight Closure with exactly m nodes, we can use:
    # For a given λ, solve: maximize sum_{i in S} w_i where w_i = deg_pre[i] - λ
    # subject to S being a closure in the comparability DAG.
    # But comparability is not transitive in a simple way for closure.
    #
    # Alternative: Use the fact that for a fixed set S, the number of comparable pairs is:
    # (1/2) * sum_{i in S} (number of j in S comparable to i)
    # = (1/2) * sum_{i in S} |S ∩ N(i)|
    #
    # Let's use a different approach: direct optimization with simulated annealing or
    # a smarter local search using the precomputed bitsets.
    
    # Initialize with a good starting solution: greedy
    current = []
    current_set = set()
    
    # Start with a chain
    chain_mask = 0
    for i in range(n):
        chain_mask |= (1 << i)
        current.append(chain_mask)
        current_set.add(chain_mask)
    
    # Greedily add remaining
    while len(current) < m:
        if time.time() > deadline - 15: break
        
        best_x = -1
        best_score = -1
        for x in range(total_masks):
            if x in current_set: continue
            # Count how many in current are comparable to x
            score = popcount(comp_bits[x] & ((1 << total_masks) - 1))
            # Actually, we need to count only those in current_set
            # Let's build a bitset for current_set
            # This is inefficient; let's just loop
            score = 0
            for y in current:
                if is_comparable(x, y):
                    score += 1
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x != -1:
            current.append(best_x)
            current_set.add(best_x)
        else:
            break
    
    # Calculate current score
    def calc_score(sets):
        s = 0
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                s += is_comparable(sets[i], sets[j])
        return s
    
    current_score = calc_score(current)
    
    # Local search with bitset acceleration
    # Maintain a bitset for the current set
    current_bitset = 0
    for mask in current:
        current_bitset |= (1 << mask)
    
    # deg[i] = number of elements in current comparable to i
    deg = [0] * total_masks
    for y in range(total_masks):
        deg[y] = popcount(comp_bits[y] & current_bitset)
    
    # Simulated annealing / hill climbing
    improved = True
    temperature = 5.0
    min_temperature = 0.1
    cooling_rate = 0.99
    
    while improved and time.time() < deadline - 5:
        improved = False
        best_delta = 0
        best_i = -1
        best_x = -1
        
        for i in range(len(current)):
            c_i = current[i]
            deg_c_i = deg[c_i]
            
            for x in range(total_masks):
                if x in current_set: continue
                
                comp_x_c = 1 if is_comparable(x, c_i) else 0
                delta = (deg[x] - comp_x_c) - deg_c_i
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_x = x
        
        if best_i != -1 and best_delta > 0:
            old_c = current[best_i]
            current[best_i] = best_x
            current_set.remove(old_c)
            current_set.add(best_x)
            
            # Update bitset
            current_bitset &= ~(1 << old_c)
            current_bitset |= (1 << best_x)
            
            # Update deg
            for y in range(total_masks):
                comp_y_new = 1 if is_comparable(y, best_x) else 0
                comp_y_old = 1 if is_comparable(y, old_c) else 0
                diff = comp_y_new - comp_y_old
                if diff != 0:
                    deg[y] += diff
            
            current_score += best_delta
            improved = True
            temperature *= cooling_rate
        else:
            # Try random perturbation if stuck
            if temperature > min_temperature and time.time() < deadline - 5:
                import random
                for _ in range(100):
                    i = random.randint(0, len(current) - 1)
                    x = random.randint(0, total_masks - 1)
                    if x in current_set: continue
                    
                    old_c = current[i]
                    comp_x_c = 1 if is_comparable(x, old_c) else 0
                    delta = (deg[x] - comp_x_c) - deg[old_c]
                    
                    # Accept with probability exp(delta/temperature) if delta < 0
                    if delta >= 0 or (random.random() < pow(2.718281828, delta / temperature)):
                        current[i] = x
                        current_set.remove(old_c)
                        current_set.add(x)
                        
                        current_bitset &= ~(1 << old_c)
                        current_bitset |= (1 << x)
                        
                        for y in range(total_masks):
                            comp_y_new = 1 if is_comparable(y, x) else 0
                            comp_y_old = 1 if is_comparable(y, old_c) else 0
                            diff = comp_y_new - comp_y_old
                            if diff != 0:
                                deg[y] += diff
                        
                        current_score += delta
                        improved = True
                        break
                temperature *= cooling_rate
    
    # Convert masks to lists of integers
    result = []
    for mask in current:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)
        
    return result
# EVOLVE-BLOCK-END
