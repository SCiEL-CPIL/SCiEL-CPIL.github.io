import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # Safety margin before 90s kill

def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    # BFS from FULL_MASK
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        cur = queue.popleft()
        
        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0

def _generate_initial_candidates():
    """Generate a diverse set of initial candidates including known high-threshold structures."""
    candidates = []
    
    # 1. Standard Ping-Pong structure variations
    # Theoretical max is (N-1)^2 = 121.
    # A known structure achieving this:
    # a: 0->1->...->N-1->0
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    a_cycle = [(i + 1) % N for i in range(N)]
    b_pingpong = [0] * N
    b_pingpong[0] = 0
    b_pingpong[1] = 0
    for i in range(2, N):
        b_pingpong[i] = i - 1
    candidates.append((a_cycle[:], b_pingpong[:]))
    
    # Variant: b maps N-1 to 0 instead of N-2?
    b_var1 = [0] * N
    b_var1[0] = 0
    b_var1[1] = 0
    for i in range(2, N-1):
        b_var1[i] = i - 1
    b_var1[N-1] = 0
    candidates.append((a_cycle[:], b_var1[:]))
    
    # Variant: b is a different permutation
    # a: cycle
    # b: 0->0, 1->1, 2->3, 3->4, ... N-1->0?
    b_var2 = [0] * N
    b_var2[0] = 0
    b_var2[1] = 1
    for i in range(2, N):
        b_var2[i] = (i + 1) % N
    candidates.append((a_cycle[:], b_var2[:]))

    # 2. Random permutations
    for _ in range(20):
        a_rand = list(range(N))
        random.shuffle(a_rand)
        b_rand = list(range(N))
        random.shuffle(b_rand)
        candidates.append((a_rand, b_rand))
        
    # 3. Random functions (not just permutations)
    for _ in range(20):
        a_rand = [random.randint(0, N-1) for _ in range(N)]
        b_rand = [random.randint(0, N-1) for _ in range(N)]
        candidates.append((a_rand, b_rand))
        
    return candidates

def _local_search(a, b, start_time, deadline, max_iterations=5000):
    """
    Hill climbing local search to maximize the threshold.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    
    if cur_score == 0:
        return None, None, 0
        
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    no_improve = 0
    max_no_improve = 200  # Increase patience to escape local optima like 112
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose a random state and a random letter to mutate
        state = random.randint(0, N-1)
        if random.random() < 0.5:
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[state] = old_dest
                no_improve += 1
        else:
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_b[state] = old_dest
                no_improve += 1
                
        # Occasionally perform a larger jump (swap two destinations) to escape local minima
        if no_improve > 50 and random.random() < 0.1:
            i, j = random.sample(range(N), 2)
            if random.random() < 0.5:
                cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
            else:
                cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                # Revert
                if random.random() < 0.5:
                    cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                else:
                    cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
                no_improve += 10 # Penalize failed jumps more heavily

    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    random.seed(42)
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Evaluate initial candidates
    candidates = _generate_initial_candidates()
    
    for a_cand, b_cand in candidates:
        if time.time() > deadline - 10.0:
            break
        score = compute_threshold(a_cand, b_cand)
        if score > best_score:
            best_score = score
            best_a = a_cand[:]
            best_b = b_cand[:]
            
    # If we already hit the theoretical max, we can return early
    if best_score == (N-1)**2:
        return [best_a, best_b]

    # Phase 2: Local search from the best candidate
    if best_a is not None:
        refined_a, refined_b, refined_score = _local_search(
            best_a, best_b, start_time, deadline - 5.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b
            
    if best_score == (N-1)**2:
        return [best_a, best_b]

    # Phase 3: Multiple random restarts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        num_restarts = min(50, int(remaining_time / 2.0))
        for i in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            a_rand = [random.randint(0, N-1) for _ in range(N)]
            b_rand = [random.randint(0, N-1) for _ in range(N)]
            
            # Quick check
            score = compute_threshold(a_rand, b_rand)
            if score == 0:
                continue
                
            # If promising, refine
            if score >= best_score - 10:
                refined_a, refined_b, refined_score = _local_search(
                    a_rand, b_rand, start_time, deadline - 2.0, max_iterations=2000
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b
                    
            if best_score == (N-1)**2:
                return [best_a, best_b]
                
    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
    
    return [best_a, best_b]

# Optional: Test locally
if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
