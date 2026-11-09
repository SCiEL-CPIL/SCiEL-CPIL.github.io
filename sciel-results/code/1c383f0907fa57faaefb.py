import itertools
import time
import random
import sys
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=6, radius=1
    # We use a simulated annealing approach starting from a direct-sum construction.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (sphere of radius 1)
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = [word_to_idx[w]]  # include self
        for i in range(n):
            orig_val = w_list[i]
            for v in range(q):
                if v != orig_val:
                    w_list[i] = v
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(nbrs)
    
    # Convert neighbors to sets for fast membership testing
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # Initial code: direct sum of tetracode
    # Tetracode: 9 words of length 4 that perfectly cover {0,1,2}^4 with radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Build initial code by combining tetracode with all tails of length n-4
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(cw)] for cw in initial_code]
    
    # Compute coverage counts
    coverage = [0] * total_words
    for ci in code_indices:
        for nbr in neighbors[ci]:
            coverage[nbr] += 1
    
    # Count uncovered words
    uncovered_count = sum(1 for c in coverage if c == 0)
    
    # Simulated Annealing
    best_code = list(code_indices)
    best_uncovered = uncovered_count
    best_size = len(code_indices)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Check if we've reached a perfect code
        if best_uncovered == 0 and best_size <= 73:
            break
        
        # Random move: pick a random codeword and either remove it or add a new one
        move_type = rng.random()
        
        if move_type < 0.3 and len(code_indices) > 1:
            # Remove a random codeword
            idx_to_remove = rng.randrange(len(code_indices))
            ci = code_indices[idx_to_remove]
            
            # Calculate change in uncovered
            # A word becomes uncovered if its coverage drops to 0
            delta_uncovered = 0
            for nbr in neighbors[ci]:
                if coverage[nbr] == 1:
                    delta_uncovered += 1
            
            # Apply move
            for nbr in neighbors[ci]:
                coverage[nbr] -= 1
            code_indices.pop(idx_to_remove)
            uncovered_count += delta_uncovered
            
            # Accept or reject
            if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(code_indices) < best_size):
                best_uncovered = uncovered_count
                best_size = len(code_indices)
                best_code = list(code_indices)
            elif uncovered_count <= best_uncovered:
                # Accept with probability based on temperature
                delta = uncovered_count - best_uncovered
                if delta > 0 and rng.random() > math.exp(-delta / T):
                    best_uncovered = uncovered_count
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                else:
                    # Reject
                    for nbr in neighbors[ci]:
                        coverage[nbr] += 1
                    code_indices.insert(idx_to_remove, ci)
                    uncovered_count -= delta_uncovered
            else:
                # Reject
                for nbr in neighbors[ci]:
                    coverage[nbr] += 1
                code_indices.insert(idx_to_remove, ci)
                uncovered_count -= delta_uncovered
            
        elif move_type < 0.5:
            # Add a random codeword - prefer words that cover many uncovered words
            # Find uncovered words
            uncovered_words = [i for i in range(total_words) if coverage[i] == 0]
            
            if uncovered_words:
                # Pick a random uncovered word and add a codeword that covers it
                target_word = rng.choice(uncovered_words)
                # Find all codewords that would cover this word
                # A codeword ci covers target_word if target_word is in neighbors[ci]
                # So we need ci such that target_word in neighbors[ci]
                # This means ci is within distance 1 of target_word
                candidate_codewords = neighbors[target_word]  # all words within distance 1 of target_word
                new_ci = rng.choice(candidate_codewords)
            else:
                new_ci = rng.randrange(total_words)
            
            # Check if it's already in the code
            if new_ci in code_indices:
                continue
            
            # Calculate change in uncovered
            # A word becomes covered if its coverage was 0
            delta_uncovered = 0
            for nbr in neighbors[new_ci]:
                if coverage[nbr] == 0:
                    delta_uncovered -= 1  # uncovered decreases
            
            # Apply move
            for nbr in neighbors[new_ci]:
                coverage[nbr] += 1
            code_indices.append(new_ci)
            uncovered_count += delta_uncovered
            
            # Accept or reject
            if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(code_indices) < best_size):
                best_uncovered = uncovered_count
                best_size = len(code_indices)
                best_code = list(code_indices)
            elif uncovered_count <= best_uncovered:
                delta = uncovered_count - best_uncovered
                if delta > 0 and rng.random() > math.exp(-delta / T):
                    best_uncovered = uncovered_count
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                else:
                    # Reject
                    for nbr in neighbors[new_ci]:
                        coverage[nbr] -= 1
                    code_indices.pop()
                    uncovered_count -= delta_uncovered
            else:
                # Reject
                for nbr in neighbors[new_ci]:
                    coverage[nbr] -= 1
                code_indices.pop()
                uncovered_count -= delta_uncovered
        
        else:
            # Swap: remove one and add another
            if len(code_indices) < 1:
                continue
            idx_to_remove = rng.randrange(len(code_indices))
            ci_remove = code_indices[idx_to_remove]
            
            # Find uncovered words
            uncovered_words = [i for i in range(total_words) if coverage[i] == 0]
            
            if uncovered_words:
                # Pick a random uncovered word and find a codeword that covers it
                target_word = rng.choice(uncovered_words)
                candidate_codewords = neighbors[target_word]
                new_ci = rng.choice(candidate_codewords)
            else:
                new_ci = rng.randrange(total_words)
            
            if new_ci in code_indices:
                continue
            
            # Calculate combined change
            # First, what happens if we remove ci_remove
            delta_remove = 0
            nbr_set_remove = neighbor_sets[ci_remove]
            for nbr in neighbors[ci_remove]:
                if coverage[nbr] == 1:
                    delta_remove += 1
            
            # Then, what happens if we add new_ci (after removal)
            delta_add = 0
            for nbr in neighbors[new_ci]:
                # After removal, coverage[nbr] is coverage[nbr] - (1 if nbr in neighbors[ci_remove] else 0)
                effective_cov = coverage[nbr]
                if nbr in nbr_set_remove:
                    effective_cov -= 1
                if effective_cov == 0:
                    delta_add -= 1
            
            total_delta = delta_remove + delta_add
            
            # Apply move
            for nbr in neighbors[ci_remove]:
                coverage[nbr] -= 1
            for nbr in neighbors[new_ci]:
                coverage[nbr] += 1
            code_indices[idx_to_remove] = new_ci
            uncovered_count += total_delta
            
            # Accept or reject
            if uncovered_count < best_uncovered or (uncovered_count == best_uncovered and len(code_indices) < best_size):
                best_uncovered = uncovered_count
                best_size = len(code_indices)
                best_code = list(code_indices)
            elif uncovered_count <= best_uncovered:
                delta = uncovered_count - best_uncovered
                if delta > 0 and rng.random() > math.exp(-delta / T):
                    best_uncovered = uncovered_count
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                else:
                    # Reject
                    for nbr in neighbors[new_ci]:
                        coverage[nbr] -= 1
                    for nbr in neighbors[ci_remove]:
                        coverage[nbr] += 1
                    code_indices[idx_to_remove] = ci_remove
                    uncovered_count -= total_delta
            else:
                # Reject
                for nbr in neighbors[new_ci]:
                    coverage[nbr] -= 1
                for nbr in neighbors[ci_remove]:
                    coverage[nbr] += 1
                code_indices[idx_to_remove] = ci_remove
                uncovered_count -= total_delta
        
        # Cool down
        T = max(T * cooling, T_min)
    
    # Final polish: try to remove redundant codewords
    final_code = list(best_code)
    if time.time() < deadline - 5.0:
        # Recompute coverage for final_code
        final_coverage = [0] * total_words
        for ci in final_code:
            for nbr in neighbors[ci]:
                final_coverage[nbr] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(final_code) - 1, -1, -1):
                ci = final_code[i]
                is_redundant = True
                for nbr in neighbors[ci]:
                    if final_coverage[nbr] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    final_code.pop(i)
                    for nbr in neighbors[ci]:
                        final_coverage[nbr] -= 1
                    changed = True
                    break
    
    # Convert back to word lists
    result = [list(all_words[i]) for i in final_code]
    
    # Verify validity
    covered = set()
    for cw in result:
        idx = word_to_idx[tuple(cw)]
        covered.update(neighbors[idx])
    
    if len(covered) != total_words:
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return result
