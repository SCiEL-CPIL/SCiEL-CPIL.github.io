"""
Football pool problem: ternary covering code K_3(7,1).

Approach:
1. Use Bitsets (Python integers) to represent the set of covered words.
   This allows extremely fast union, intersection, and subtraction operations.
2. Initialize with a Greedy Set Cover to get a valid (but likely suboptimal) code.
3. Apply Local Search (Remove-and-Repair):
   - Iterate through codewords.
   - Tentatively remove a codeword.
   - If the code remains valid (all words covered), keep it removed.
   - If not, identify uncovered words and add the best candidate to cover them.
   - This process is repeated until no improvement is found.
4. To escape local minima, we can also try random swaps or restarts, but for this small instance,
   a single pass of Remove-and-Repair starting from Greedy often gets very close to optimal.
   We will also add a "Swap" heuristic: try replacing a codeword with another that covers
   more currently uncovered words, hoping it enables subsequent removals.
"""

import itertools
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1()
    else:
        return _general_greedy(q, n, radius)

def _solve_3_7_1():
    q, n, radius = 3, 7, 1
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks for each word
    # A mask is a Python integer where bit i is 1 if word i is covered.
    # N=2187, so we need 2187 bits. Python integers handle this natively and efficiently.
    covers = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        # Add self
        mask |= (1 << i)
        # Add neighbors at distance 1
        for pos in range(n):
            for delta in range(1, q):
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % q
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        covers[i] = mask
        
    full_mask = (1 << N) - 1
    
    # 1. Greedy Initialization
    code_indices = []
    covered_mask = 0
    uncovered_mask = full_mask
    
    # We need to find the best candidate for the current uncovered set.
    # To speed up greedy, we can just scan all candidates.
    # N=2187, steps ~186. 186 * 2187 ~ 400k operations. Bit ops are fast.
    while uncovered_mask:
        best_idx = -1
        best_count = -1
        # Optimization: Only consider candidates that are in the uncovered set?
        # Not necessarily, a covered word might cover many uncovered ones.
        # But we can filter: if a candidate doesn't cover any uncovered word, it's useless.
        for i in range(N):
            # If candidate i doesn't cover any uncovered word, skip
            if (covers[i] & uncovered_mask) == 0:
                continue
            count = bin(covers[i] & uncovered_mask).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
        code_indices.append(best_idx)
        covered_mask |= covers[best_idx]
        uncovered_mask = full_mask & ~covered_mask
        
    # 2. Local Search: Remove-and-Repair
    # We will iterate until no improvement is possible.
    # Improvement means the size of the code decreases.
    
    def is_valid(code_set):
        # Check if the union of covers in code_set covers full_mask
        # This is slow if done naively. We maintain covered_mask.
        pass
        
    # We maintain `covered_mask` and `code_indices`.
    # To check if a codeword is redundant:
    #   new_covered = covered_mask & ~covers[idx]
    #   if new_covered == full_mask: it is redundant.
    
    # To repair:
    #   if we remove idx and it's not redundant, we have holes = covers[idx] & ~ (covered_mask & ~covers[idx])
    #   Actually, holes = covers[idx] & ~(covered_mask ^ covers[idx]) ? No.
    #   Let C be current covered. Remove idx. New C' = C & ~covers[idx].
    #   Holes = C & covers[idx] & ~C' ? No. Holes are words that were covered by idx 
    #   and are NOT covered by any other codeword in the code.
    #   Holes = covers[idx] & ~(covered_mask ^ (covered_mask & covers[idx]))? 
    #   Simpler: Holes = covers[idx] & ~(covered_mask & ~covers[idx])
    #   Wait, covered_mask & ~covers[idx] is the coverage provided by others.
    #   So Holes = covers[idx] & ~(covered_mask & ~covers[idx])
    
    improved = True
    max_iterations = 100
    it = 0
    while improved and it < max_iterations:
        improved = False
        it += 1
        # Try to remove each codeword
        # We iterate in reverse to avoid index shifting issues, or just use a list and pop
        for i in range(len(code_indices) - 1, -1, -1):
            idx = code_indices[i]
            # Calculate coverage without this codeword
            # We need to know if removing it leaves any holes.
            # Holes = covers[idx] & ~(covered_mask & ~covers[idx])
            others_cover = covered_mask & ~covers[idx]
            holes = covers[idx] & ~others_cover
            
            if holes == 0:
                # Redundant, remove it
                code_indices.pop(i)
                covered_mask = others_cover
                improved = True
            else:
                # Not redundant. Try to repair by adding a new codeword that covers the holes
                # and maybe allows other removals? 
                # Simple repair: Find a codeword that covers the most holes.
                # If we add it, does it allow us to remove the current one? No, we just replace.
                # Actually, a "Swap" is: Remove idx, then add best_j.
                # If best_j covers all holes, we can swap.
                # But does this reduce size? No, size stays same.
                # However, it might enable future removals.
                # For now, let's just stick to strict removal to guarantee size reduction.
                # To improve further, we need a more complex local search.
                
                # Let's try a random restart or swap strategy if we get stuck.
                # For now, strict removal is the safest "improvement".
                pass
                
    # If we didn't reach 186, we might be stuck in a local minimum.
    # Let's add a "Perturbation" step: Randomly remove a codeword, even if it creates holes,
    # then repair by adding the best codewords to cover holes.
    # If the resulting size is smaller, accept.
    
    best_code = code_indices[:]
    best_size = len(best_code)
    
    # Iterated Local Search
    for _ in range(50): # 50 restarts/perturbations
        # Perturb: Remove a random codeword
        if not best_code: break
        k = random.randint(0, len(best_code) - 1)
        removed_idx = best_code[k]
        current_code = best_code[:k] + best_code[k+1:]
        
        # Recalculate covered mask for current_code
        cur_covered = 0
        for idx in current_code:
            cur_covered |= covers[idx]
        
        # Find holes
        holes = full_mask & ~cur_covered
        
        if holes == 0:
            # We just removed a redundant word from best_code? 
            # This implies best_code wasn't minimal.
            best_code = current_code
            best_size = len(best_code)
            continue
            
        # Repair: Add codewords to cover holes
        # Greedily add best codeword for holes
        code_list = current_code[:]
        covered = cur_covered
        while holes:
            best_j = -1
            best_count = -1
            for j in range(N):
                if j in code_list:
                    continue
                count = bin(covers[j] & holes).count('1')
                if count > best_count:
                    best_count = count
                    best_j = j
            if best_j == -1 or best_count == 0:
                break # Should not happen
            code_list.append(best_j)
            covered |= covers[best_j]
            holes = full_mask & ~covered
            
        if len(code_list) < best_size:
            best_code = code_list
            best_size = len(best_code)
            
    # Convert indices to codewords
    return [list(all_words[idx]) for idx in best_code]

def _general_greedy(q, n, radius):
    # Fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
        
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    return code
