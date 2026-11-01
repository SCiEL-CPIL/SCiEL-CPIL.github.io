import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return solve_3_6_1()
    # Fallback: return all words (valid but large)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def solve_3_6_1():
    q = 3
    n = 6
    N = q ** n  # 729
    
    # Precompute sphere indices using base-3 arithmetic (fast)
    # Sphere size = 1 + n*(q-1) = 13
    sphere_indices = np.zeros((N, 13), dtype=np.int32)
    for i in range(N):
        # Convert index i to base-3 word
        w = np.zeros(n, dtype=np.int32)
        tmp = i
        for pos in range(n - 1, -1, -1):
            w[pos] = tmp % 3
            tmp //= 3
        
        sphere_indices[i, 0] = i
        idx = 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx_val = 0
                for j in range(n):
                    idx_val = idx_val * 3 + new_w[j]
                sphere_indices[i, idx] = idx_val
                idx += 1
    
    # Build coverage matrix using boolean OR for efficiency
    # cov[c] is a boolean vector of length N
    cov = np.zeros((N, N), dtype=np.bool_)
    for c in range(N):
        cov[c, sphere_indices[c]] = True
    
    # Precompute all_words for output
    all_words = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        tmp = i
        for pos in range(n - 1, -1, -1):
            all_words[i, pos] = tmp % 3
            tmp //= 3
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Phase 1: Greedy set cover for initial solution
    def greedy_cover():
        is_covered = np.zeros(N, dtype=np.bool_)
        selected = []
        while not np.all(is_covered):
            # Gain: number of uncovered points each center covers
            gain = cov[:, ~is_covered].sum(axis=1)
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            c = random.choice(candidates.tolist())
            selected.append(c)
            is_covered |= cov[c]
        return selected
    
    # Multiple random greedy restarts
    best_solution = None
    best_size = N
    
    for restart in range(20):
        if time.time() > deadline - 60:
            break
        sol = greedy_cover()
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
    
    # Phase 2: Local search with swaps and removals
    if best_solution is not None:
        current = list(best_solution)
        current_set = set(current)
        
        # Compute coverage counts
        coverage_count = np.zeros(N, dtype=np.int32)
        for c in current:
            coverage_count += cov[c]
        
        # Try to remove redundant codewords
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            for i in range(len(current)):
                c = current[i]
                # Check if removing c leaves any point uncovered
                if np.any(coverage_count[sphere_indices[c]] == 1):
                    continue
                # Remove c
                current.pop(i)
                coverage_count -= cov[c]
                best_size = len(current)
                best_solution = current[:]
                improved = True
                break
        
        # Swap-based local search
        # Try replacing one codeword with another to reduce size or maintain
        # Focus: find a codeword whose removal + addition of a better one reduces count
        # Actually, for minimization, we want to remove without adding.
        # But sometimes a swap (remove one, add one) enables further removals.
        
        # More aggressive: 1-1 exchange to find better configurations
        for it in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Pick a random current codeword to potentially remove
            c_remove = random.choice(current)
            
            # Tentatively remove it
            coverage_count -= cov[c_remove]
            uncovered_after_remove = np.sum(coverage_count == 0)
            
            if uncovered_after_remove == 0:
                # Great! We can just remove it
                current.remove(c_remove)
                best_size = len(current)
                best_solution = current[:]
                continue
            
            # Try to find a codeword to add that covers more uncovered points than c_remove did
            # Actually, we want to find c_add such that after adding, we can remove c_remove
            # and still be covered, OR we find a c_add that covers the uncovered points
            # and allows removal of c_remove.
            
            # Simpler: try adding a random codeword, then try removing c_remove
            # If we can remove c_remove after adding c_add, it's a valid swap (same size)
            # But we want to eventually reduce size.
            
            # Let's try: for each uncovered point after removing c_remove,
            # find codewords that cover it, add one, then try to remove c_remove.
            
            uncovered_points = np.where(coverage_count == 0)[0]
            if len(uncovered_points) == 0:
                continue
            
            # Pick a random uncovered point
            p = random.choice(uncovered_points.tolist())
            
            # Find codewords that cover p
            # This is expensive to compute on the fly. Let's precompute reverse index.
            # Actually, cov is (N, N). cov[c, p] is True if c covers p.
            # So candidates_add = np.where(cov[:, p])[0]
            candidates_add = np.where(cov[:, p])[0]
            
            # Exclude c_remove if it's in candidates (it shouldn't be, since it was removed and p is uncovered)
            # Actually c_remove covers p (since p was covered before removal), so c_remove is in candidates_add
            # But we just removed it, so we're looking for a replacement.
            # We want to add c_add, then remove c_remove. Net change: 0 in size.
            # But maybe after adding c_add, we can remove c_remove AND some other codeword?
            
            # Let's just do a simple swap: add c_add, remove c_remove.
            # This maintains size but might enable future removals.
            c_add = random.choice(candidates_add.tolist())
            if c_add == c_remove:
                continue
            
            # Add c_add
            coverage_count += cov[c_add]
            current.append(c_add)
            
            # Now try to remove c_remove
            coverage_count -= cov[c_remove]
            if np.all(coverage_count > 0):
                # Can remove c_remove
                current.remove(c_remove)
                # Size reduced by 1
                best_size = len(current)
                best_solution = current[:]
                # Reset coverage_count since we removed c_remove
                # Actually we already did coverage_count -= cov[c_remove]
                # So current state is: c_add added, c_remove removed
                continue
            else:
                # Cannot remove c_remove, revert
                coverage_count += cov[c_remove]
                current.remove(c_add)
    
    # Phase 3: Final cleanup - remove any redundant codewords
    if best_solution is not None:
        current = list(best_solution)
        coverage_count = np.zeros(N, dtype=np.int32)
        for c in current:
            coverage_count += cov[c]
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                c = current[i]
                if np.any(coverage_count[sphere_indices[c]] == 1):
                    continue
                current.pop(i)
                coverage_count -= cov[c]
                improved = True
                break
        
        if len(current) < best_size:
            best_size = len(current)
            best_solution = current[:]
    
    # Convert to list of lists
    if best_solution is None:
        # Fallback: use a known construction
        # Direct sum of tetracode and length-2 code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Length-2 radius-1 code over q=3: need to cover 9 words with spheres of size 1+2*2=5
        # Minimum is ceil(9/5) = 2. Use words [0,0] and [1,1]? Let's check.
        # Actually, for n=2, q=3, r=1, K=2 is optimal.
        # Words: [0,0] covers [0,0],[1,0],[2,0],[0,1],[0,2]
        # [1,1] covers [1,1],[0,1],[2,1],[1,0],[1,2]
        # Union: [0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[2,1],[1,2]
        # Missing: [2,2]. So need a third word? [2,2] covers [2,2],[1,2],[0,2],[2,1],[2,0]
        # So 3 words suffice. Actually K_3(2,1) = 3.
        # Let's use [0,0], [1,1], [2,2]
        code_2 = [[0,0], [1,1], [2,2]]
        best_solution = [h + t for h in tetracode for t in code_2]
    
    return [all_words[c].tolist() for c in best_solution]
