import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 5s margin before 120s kill
    
    if n <= 8 and radius == 1:
        total_words = q ** n
        sphere_size = 1 + n * (q - 1)
        
        # Precompute all neighbor indices for each word
        # neighbors[i] = list of indices within Hamming distance 1 of word i
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs
        
        # Build inverse neighbor map: for each word, which codewords cover it
        # inv_neighbors[j] = list of codeword indices i such that j is in neighbors[i]
        # This allows O(1) lookup of which codewords cover a given word
        
        # Greedy: pick word covering most uncovered
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while time.time() < deadline - 35.0:  # Leave 35s for local search
            gains = (~covered)[neighbors].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            code_indices.append(best_idx)
            covered[neighbors[best_idx]] = True
        
        # Helper: compute coverage count vector from a list of codewords
        def compute_coverage(code_list):
            """Returns array where coverage[j] = number of codewords covering word j."""
            if len(code_list) == 0:
                return np.zeros(total_words, dtype=np.int32)
            cov = np.zeros(total_words, dtype=np.int32)
            for ci in code_list:
                cov[neighbors[ci]] += 1
            return cov
        
        # Helper: find redundant codeword using coverage counts
        def find_redundant(code_list, cov):
            """Returns index of a redundant codeword in code_list, or -1 if none."""
            k = len(code_list)
            if k == 0:
                return -1
            # For each codeword, check if all its neighbors are covered by >= 2 codewords
            # A codeword ci is redundant if min(cov[j] for j in neighbors[ci]) >= 2
            # (since removing ci decreases cov[j] by 1 for j in neighbors[ci])
            for idx, ci in enumerate(code_list):
                nbrs = neighbors[ci]
                if np.min(cov[nbrs]) >= 2:
                    return idx
            return -1
        
        # Local search: try to remove redundant codewords using vectorized coverage
        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            cov = compute_coverage(code_indices)
            idx = find_redundant(code_indices, cov)
            if idx >= 0:
                code_indices.pop(idx)
                changed = True
        
        # Add-then-remove heuristic: try adding a codeword, then removing two to see if we can reduce size
        # This can escape local minima of the simple removal heuristic
        # We only accept the change if we successfully remove 2 codewords (net -1 size)
        while time.time() < deadline - 8.0:
            # Try random add-then-remove
            if len(code_indices) < 2:
                break
            
            # Pick a random candidate to add
            candidate = random.randint(0, total_words - 1)
            if candidate in code_indices:
                continue
            
            # Save current state to allow rollback
            old_code = list(code_indices)
            
            # Temporarily add it
            code_indices.append(candidate)
            
            # Compute coverage with the new codeword
            cov = compute_coverage(code_indices)
            
            # Now try to remove two codewords (including possibly the new one)
            # to see if we can end up with fewer total
            removed = 0
            for _ in range(2):
                if time.time() > deadline - 8.0:
                    break
                # Find a redundant codeword to remove
                idx = find_redundant(code_indices, cov)
                if idx < 0:
                    break
                code_indices.pop(idx)
                # Update coverage
                cov = compute_coverage(code_indices)
                removed += 1
            
            # We only keep the change if we removed 2 codewords (net -1)
            # If removed < 2, revert to old state
            if removed < 2:
                code_indices = old_code
        
        # Final pass: remove any remaining redundant codewords
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            cov = compute_coverage(code_indices)
            idx = find_redundant(code_indices, cov)
            if idx >= 0:
                code_indices.pop(idx)
                changed = True
        
        # Convert to words
        result_words = []
        for idx in code_indices:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result_words.append(w)
        
        return result_words
    else:
        # Fallback: direct sum with tetracode-like base
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            base_n = 4
            base_code = tetracode
        else:
            base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            base_n = 4
        
        if n < base_n:
            return [list(range(n))]
        
        tails = list(itertools.product(range(q), repeat=n - base_n))
        return [h + list(tail) for h in base_code for tail in tails]
