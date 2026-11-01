import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as bitmasks
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask
        
    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 110.0 # 40s margin for safety
    
    best_code = None
    best_size = N # worst case
    
    # Helper to prune a code
    def prune(code_list):
        current = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                # Check if current[i] is redundant
                union_others = 0
                # Optimization: if len(current) is small, this is fast.
                # We can break early if union_others becomes full_mask
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    # Redundant, remove it
                    current.pop(i)
                    changed = True
                    break # Restart the for loop
        return current

    # Helper to perform local search swaps
    def local_search(code_list):
        current = list(code_list)
        current_set = set(current)
        union_mask = 0
        for i in current:
            union_mask |= spheres[i]
            
        # Try to swap codewords to reduce size or improve coverage
        # Since we start with a full cover, we just try to find redundant ones or swaps that allow removal of others?
        # Actually, standard local search for set cover:
        # 1. Remove redundant
        # 2. Try to swap a codeword for another that covers same unique part but adds more coverage? 
        #    Or simply: try to replace a codeword with one that allows removing another?
        # 
        # The prompt's "local search" in the current program was:
        # "Second pass: try swaps to improve coverage"
        # It checks if unique_i is covered by j, and if j adds new coverage.
        # This effectively tries to move the "center" of coverage to a spot that is at least as good and potentially better.
        
        improved = True
        while improved:
            improved = False
            for i in current:
                others_mask = 0
                for j in current:
                    if j != i:
                        others_mask |= spheres[j]
                unique_i = spheres[i] & ~others_mask
                
                if unique_i == 0:
                    continue
                
                best_j = -1
                best_gain = -1
                for j in range(N):
                    if j in current_set:
                        continue
                    # j must cover everything i uniquely covers
                    if (unique_i & ~spheres[j]) != 0:
                        continue
                    # j should add new coverage
                    gain = (spheres[j] & ~union_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                
                if best_j != -1 and best_gain > 0:
                    current.remove(i)
                    current_set.discard(i)
                    current.append(best_j)
                    current_set.add(best_j)
                    union_mask = (union_mask & ~spheres[i]) | spheres[best_j]
                    improved = True
                    break
        
        return current

    # Initial strategy: Start from Tetracode product (size 81) and prune/search
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])
            
    pruned_init = prune(init_code)
    ls_init = local_search(pruned_init)
    if len(ls_init) < best_size:
        best_size = len(ls_init)
        best_code = [all_words[i] for i in ls_init]
        
    # Now run greedy restarts
    # Greedy tends to find slightly different structures.
    num_restarts = 5000 # High number, we will break on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        
        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                break
                
            chosen.append(best_idx)
            covered |= spheres[best_idx]
            
        # Prune the greedy result
        pruned_greedy = prune(chosen)
        
        # Apply local search to the pruned result
        ls_greedy = local_search(pruned_greedy)
        
        if len(ls_greedy) < best_size:
            best_size = len(ls_greedy)
            best_code = [all_words[i] for i in ls_greedy]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]
