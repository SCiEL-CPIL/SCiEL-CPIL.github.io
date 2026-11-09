"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (deletion/swap) to minimize size.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Known optimal is 186. We try to find a small code.
        # Start with a reasonable initial solution.
        # The direct product construction gives 9 * 3^3 = 243.
        # We can use a better initial seed or just run the optimizer on the 243 set.
        
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        
        # Precompute neighbors (distance <= radius) for each word in the space
        # Since n=7, q=3, radius=1, each word has 1 + 7*2 = 15 neighbors.
        # We can represent the set of covered words as a bitset (integer or list of bools)
        # 2187 bits is small enough to fit in a few integers or a Python int.
        
        # Map word to index
        word_to_idx = {}
        for i, w in enumerate(all_words):
            word_to_idx[w] = i
            
        # Precompute neighbors for each index
        neighbors = [set() for _ in range(len(all_words))]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].add(idx)
        
        # Initial Solution: Greedy Set Cover
        # Start with a random or structured subset.
        # Let's start with the 243 word construction and then try to remove.
        # Or better, start with a random large set and reduce?
        # Greedy Set Cover is usually a good upper bound.
        
        # Let's try starting with the 243 words from the seed program, then delete.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        codeword_indices = [word_to_idx[c] for c in initial_codewords]
        
        # Optimize: Local Search
        # Objective: Minimize number of codewords.
        # Constraint: Union of neighbors(codewords) == All Words.
        
        # We can use a bitset for covered words.
        # Python integers can act as arbitrary precision bitsets.
        covered_mask = 0
        for idx in codeword_indices:
            for nb in neighbors[idx]:
                covered_mask |= (1 << nb)
        
        full_mask = (1 << len(all_words)) - 1
        
        # Check if covered
        if covered_mask != full_mask:
             # This shouldn't happen with the seed construction, but just in case
             pass

        # Local Search Loop
        # Try to remove a codeword if it doesn't uncover any word.
        # Use perturbation-based restarts to escape local minima.
        
        best_size = len(codeword_indices)
        best_set = set(codeword_indices)
        
        def reduce_covering_code(codeword_list, time_limit):
            """Reduce a covering code by removing redundant codewords."""
            if not codeword_list:
                return codeword_list, [0] * len(all_words)
            
            current_set = list(codeword_list)
            cover_count = [0] * len(all_words)
            for idx in current_set:
                for nb in neighbors[idx]:
                    cover_count[nb] += 1
            
            removed_any = True
            while removed_any and time.time() < time_limit:
                removed_any = False
                random.shuffle(current_set)
                for idx in current_set:
                    if time.time() > time_limit:
                        break
                    # Check if redundant
                    redundant = True
                    for nb in neighbors[idx]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove idx
                        for nb in neighbors[idx]:
                            cover_count[nb] -= 1
                        current_set.remove(idx)
                        removed_any = True
            
            return current_set, cover_count
        
        # Strategy 1: Greedy Removal from the 243 set
        current_set, cover_count = reduce_covering_code(codeword_indices, deadline)
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Strategy 2: Perturbation-based local search
        # Remove random codewords, re-greedy cover, then reduce
        num_perturbations = 0
        while time.time() < deadline - 5:
            num_perturbations += 1
            
            # Start from current best set
            perturbed_set = list(best_set)
            
            # Remove k random codewords (k scales with set size)
            k = max(1, len(perturbed_set) // 10)
            random.shuffle(perturbed_set)
            removed_codewords = perturbed_set[:k]
            perturbed_set = perturbed_set[k:]
            
            # Greedily add back codewords to cover uncovered words
            # First, compute what's still covered
            cover_count = [0] * len(all_words)
            for idx in perturbed_set:
                for nb in neighbors[idx]:
                    cover_count[nb] += 1
            
            # Find uncovered words
            uncovered = [i for i in range(len(all_words)) if cover_count[i] == 0]
            
            # Greedily add codewords to cover uncovered
            greedy_added = []
            while uncovered and time.time() < deadline - 3:
                best_idx = -1
                best_count = 0
                for idx in range(len(all_words)):
                    if idx in perturbed_set or idx in greedy_added:
                        continue
                    cnt = 0
                    for nb in neighbors[idx]:
                        if cover_count[nb] == 0:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = idx
                    if cnt == len(uncovered):
                        break
                if best_idx == -1:
                    break
                greedy_added.append(best_idx)
                for nb in neighbors[best_idx]:
                    cover_count[nb] += 1
                # Update uncovered
                uncovered = [i for i in uncovered if cover_count[i] == 0]
            
            # Combine and reduce
            combined_set = perturbed_set + greedy_added
            reduced_set, _ = reduce_covering_code(combined_set, deadline - 1)
            
            if len(reduced_set) < best_size:
                best_size = len(reduced_set)
                best_set = set(reduced_set)
        
        # Return the best set found
        result = [list(all_words[idx]) for idx in best_set]
        return result

    # Generic fallback for other parameters (not used in this specific task instance but good for robustness)
    # Simple greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
