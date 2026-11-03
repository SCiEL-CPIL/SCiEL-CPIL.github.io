import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    N = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for radius 1
    neighbor_lists = [[] for _ in range(N)]
    for c_idx in range(N):
        c_word = idx_to_word(c_idx)
        neighbor_lists[c_idx].append(c_idx)
        for i in range(n):
            orig = c_word[i]
            for new_val in range(q):
                if new_val != orig:
                    new_word = c_word.copy()
                    new_word[i] = new_val
                    w_idx = word_to_idx(new_word)
                    neighbor_lists[c_idx].append(w_idx)
    
    # Precompute: for each word, which codewords cover it
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)
    
    # Build neighbor masks as bitsets for fast coverage checks
    neighbor_masks = [0] * N
    for c_idx in range(N):
        mask = 0
        for w in neighbor_lists[c_idx]:
            mask |= (1 << w)
        neighbor_masks[c_idx] = mask
    
    full_mask = (1 << N) - 1
    
    def greedy_cover_criticality():
        """Greedy that prioritizes uncovered words with fewest potential coders"""
        uncovered = full_mask
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            min_crit = N + 1
            target_word = -1
            
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                temp ^= lsb
                crit = len(word_to_coders[w_idx])
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:
                    break
            
            if target_word == -1:
                break
            
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                        
                if not has_unique:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break
                
                best_add = -1
                best_gain = -1
                
                candidate_codewords = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                for c_add in candidate_codewords:
                    covers_all = True
                    for w in newly_uncovered:
                        if (neighbor_masks[c_add] >> w) & 1 == 0:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    new_coverage = 0
                    for w in neighbor_lists[c_add]:
                        if cover_counts[w] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    
                    code.append(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    code_set.add(best_add)
                    
                    improved = True
                    break
        
        return code
    
    def simulated_annealing(code, iterations=5000):
        """Simulated annealing with delete/add/swap moves"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        U = int(np.sum(cover_counts == 0))
        best_size = len(code) if U == 0 else float('inf')
        best_code = code[:] if U == 0 else []
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9999
        P = 10000  # Penalty for uncovered words
        
        for iteration in range(iterations):
            if time.time() > deadline - 5:
                break
            
            if T < T_min:
                break
            
            r = random.random()
            
            if r < 0.4:
                # Delete
                if len(code) == 0:
                    continue
                i = random.randint(0, len(code) - 1)
                c = code[i]
                
                delta_U = 0
                for v in neighbor_lists[c]:
                    if cover_counts[v] == 1:
                        delta_U += 1
                
                delta_E = -1 + P * delta_U
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    code.pop(i)
                    code_set.discard(c)
                    for v in neighbor_lists[c]:
                        cover_counts[v] -= 1
                    U += delta_U
                    
                    if U == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
            elif r < 0.7:
                # Add
                if U > 0:
                    # Pick a random uncovered word and a random neighbor
                    unc_words = [w for w in range(N) if cover_counts[w] == 0]
                    if not unc_words:
                        continue
                    u = random.choice(unc_words)
                    w = random.choice(neighbor_lists[u])
                else:
                    w = random.randint(0, N - 1)
                
                if w in code_set:
                    continue
                
                delta_U = 0
                for v in neighbor_lists[w]:
                    if cover_counts[v] == 0:
                        delta_U -= 1
                
                delta_E = 1 + P * delta_U
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    code.append(w)
                    code_set.add(w)
                    for v in neighbor_lists[w]:
                        cover_counts[v] += 1
                    U += delta_U
                    
                    if U == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        
            else:
                # Swap
                if len(code) == 0:
                    continue
                i = random.randint(0, len(code) - 1)
                c_old = code[i]
                
                if U > 0:
                    unc_words = [w for w in range(N) if cover_counts[w] == 0]
                    if not unc_words:
                        continue
                    u = random.choice(unc_words)
                    c_new = random.choice(neighbor_lists[u])
                else:
                    c_new = random.randint(0, N - 1)
                
                if c_new == c_old or c_new in code_set:
                    continue
                
                # Simulate swap
                delta_U = 0
                for v in neighbor_lists[c_old]:
                    if cover_counts[v] == 1:
                        delta_U += 1
                for v in neighbor_lists[c_new]:
                    if cover_counts[v] == 0:
                        delta_U -= 1
                
                delta_E = P * delta_U  # Size stays same
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    code[i] = c_new
                    code_set.discard(c_old)
                    code_set.add(c_new)
                    for v in neighbor_lists[c_old]:
                        cover_counts[v] -= 1
                    for v in neighbor_lists[c_new]:
                        cover_counts[v] += 1
                    U += delta_U
                    
                    if U == 0 and len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
            
            T *= cooling
        
        return best_code if best_code else code
    
    def multilevel_coarsening():
        """
        Multi-level approach: coarsen the Hamming space by grouping positions,
        solve a smaller covering problem, then refine.
        """
        # Level 1: Coarsen by treating pairs of positions as super-positions
        # For n=8, group positions into 4 pairs: (0,1), (2,3), (4,5), (6,7)
        # Each super-position has q^2 = 9 values
        # Coarse space: 9^4 = 6561 words of length 4 over alphabet of size 9
        
        super_q = q * q  # 9
        super_n = n // 2  # 4
        
        # Map from coarse index to fine indices
        def coarse_to_fine_indices(coarse_idx):
            """Get all fine indices that map to a given coarse index"""
            # Coarse word has super_n digits, each in 0..super_q-1
            # Each coarse digit d corresponds to fine digits (d // q, d % q)
            fine_words = []
            for i in range(super_n):
                d = (coarse_idx // (super_q ** (super_n - 1 - i))) % super_q
                fine_words.append((d // q, d % q))
            
            # Generate all fine words consistent with this coarse word
            # Actually, each coarse word maps to exactly one fine word
            fine_idx = 0
            for i in range(super_n):
                d = (coarse_idx // (super_q ** (super_n - 1 - i))) % super_q
                fine_idx = fine_idx * q * q + d
            return [fine_idx]
        
        # For coarsening, we need to define a coarsened covering problem
        # A coarse codeword covers a coarse word if the corresponding fine codeword
        # covers the fine word within radius 1 in the fine space
        
        # This is complex. Let's use a simpler approach:
        # Use the coarse structure to guide the greedy selection
        
        # Alternative: Use a "block" approach where we divide positions into blocks
        # and solve covering problems on each block independently, then combine
        
        # For n=8, divide into two blocks of 4 positions each
        # Block A: positions 0-3, Block B: positions 4-7
        # Each block has q^4 = 81 words
        # A codeword covers a word if for each block, the block part is within radius 1
        # (or the total radius is 1, so at most one block can have a mismatch)
        
        # This gives us a structured way to generate good initial codewords
        
        block_size = n // 2  # 4
        block_N = q ** block_size  # 81
        
        # Precompute block neighbors
        block_neighbor_lists = [[] for _ in range(block_N)]
        for b_idx in range(block_N):
            b_word = [0] * block_size
            for i in range(block_size - 1, -1, -1):
                b_word[i] = b_idx % q
                b_idx //= q
            
            block_neighbor_lists[b_idx].append(b_idx)
            for i in range(block_size):
                orig = b_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = b_word.copy()
                        new_word[i] = new_val
                        new_idx = 0
                        for j in range(block_size):
                            new_idx = new_idx * q + new_word[j]
                        block_neighbor_lists[b_idx].append(new_idx)
        
        # Greedy on the block structure
        # State: for each fine word, track coverage
        # A fine word (a, b) where a is block A, b is block B
        # is covered by codeword (a', b') if dist(a, a') + dist(b, b') <= 1
        
        # This means: either a=a' and dist(b,b')<=1, or b=b' and dist(a,a')<=1, or a=a' and b=b'
        
        # Precompute: for each block word, its neighbors in the block
        # block_neighbors[a] = list of a' with dist(a, a') <= 1
        
        # Use this to do a structured greedy
        uncovered = full_mask
        code = []
        code_set = set()
        
        # Track which block-pairs are "active" (have codewords)
        active_pairs = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered fine word
            temp = uncovered
            lsb = temp & (-temp)
            u_idx = lsb.bit_length() - 1
            
            # Decompose u_idx into blocks
            u_word = idx_to_word(u_idx)
            a_word = u_word[:block_size]
            b_word = u_word[block_size:]
            
            a_idx = 0
            for d in a_word:
                a_idx = a_idx * q + d
            b_idx = 0
            for d in b_word:
                b_idx = b_idx * q + d
            
            # Candidate codewords: (a', b') where a' in block_neighbors[a_idx] and b' in block_neighbors[b_idx]
            # But we need dist(a, a') + dist(b, b') <= 1
            # So either: a'=a and dist(b,b')<=1, or b'=b and dist(a,a')<=1
            
            candidates = set()
            
            # Option 1: a' = a, b' in block_neighbors[b_idx]
            for b_prime in block_neighbor_lists[b_idx]:
                c_idx = a_idx * block_N + b_prime
                candidates.add(c_idx)
            
            # Option 2: b' = b, a' in block_neighbor_lists[a_idx]
            for a_prime in block_neighbor_lists[a_idx]:
                c_idx = a_prime * block_N + b_idx
                candidates.add(c_idx)
            
            # Pick the candidate that covers the most uncovered words
            best_c = -1
            best_count = -1
            
            for c_idx in candidates:
                if c_idx in code_set:
                    continue
                count = (uncovered & neighbor_masks[c_idx]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c_idx
            
            if best_c == -1:
                # Fallback: pick any candidate
                if candidates:
                    best_c = random.choice(list(candidates))
                else:
                    break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
        
        return code
    
    # Run the multilevel greedy
    best_code = multilevel_coarsening()
    best_code = local_search_remove(best_code)
    best_code = two_opt_local_search(best_code)
    
    # Apply simulated annealing
    best_code = simulated_annealing(best_code, iterations=3000)
    best_code = local_search_remove(best_code)
    best_code = two_opt_local_search(best_code)
    
    best_size = len(best_code)
    
    # Try multiple random restarts if time permits
    for restart in range(1, 10):
        if time.time() > deadline - 20:
            break
        
        random.seed(restart * 42 + 7)
        
        # Alternate between different initialization strategies
        if restart % 2 == 1:
            code = multilevel_coarsening()
        else:
            code = greedy_cover_criticality()
        
        code = local_search_remove(code)
        code = two_opt_local_search(code)
        
        # Apply simulated annealing
        code = simulated_annealing(code, iterations=2000)
        code = local_search_remove(code)
        code = two_opt_local_search(code)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != full_mask:
        uncovered = full_mask & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [idx_to_word(i) for i in best_code]
    return result
