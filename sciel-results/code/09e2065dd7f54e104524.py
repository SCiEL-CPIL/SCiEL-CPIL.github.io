import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin for 150s limit

def _time_ok():
    return time.time() < _DEADLINE

def _neighbors_of(idx, q, n):
    """Compute indices of all words within Hamming distance 1 of idx."""
    word = np.unravel_index(idx, (q,) * n)
    out = [idx]
    for pos in range(n):
        for v in range(q):
            if v == word[pos]:
                continue
            w = list(word)
            w[pos] = v
            out.append(int(np.ravel_multi_index(tuple(w), (q,) * n)))
    return out

def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 115.0

    if radius != 1:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n
    
    # Precompute neighbor sets as lists of indices
    # For q=3, n=7, total=2187. Each set has 1 + 7*2 = 15 elements.
    neighbor_sets = []
    for idx in range(total):
        neighbor_sets.append(_neighbors_of(idx, q, n))
    
    # Precompute bitsets for fast intersection/union
    # Python integers are arbitrary precision, so 2187 bits is fine.
    neighbor_bitsets = []
    for ns in neighbor_sets:
        bits = 0
        for idx in ns:
            bits |= (1 << idx)
        neighbor_bitsets.append(bits)
    
    all_bits = (1 << total) - 1

    def _remove_redundant(chosen):
        """Remove any chosen word whose coverage is fully covered by others."""
        if not _time_ok():
            return chosen
        m = len(chosen)
        if m == 0:
            return chosen
        
        changed = True
        while changed and _time_ok():
            changed = False
            for i in range(len(chosen)):
                if not _time_ok():
                    break
                # Check if chosen[i] is redundant
                others_union = 0
                for j in range(len(chosen)):
                    if j != i:
                        others_union |= neighbor_bitsets[chosen[j]]
                
                if (neighbor_bitsets[chosen[i]] & ~others_union) == 0:
                    # Redundant
                    chosen.pop(i)
                    changed = True
                    break
        return chosen

    def _randomized_greedy(seed):
        """
        Randomized greedy set cover.
        """
        rng = random.Random(seed)
        uncovered = all_bits
        chosen = []
        
        while uncovered and _time_ok():
            best_gain = 0
            candidates = []
            chosen_set = set(chosen)
            
            for c in range(total):
                if c in chosen_set:
                    continue
                gain = bin(uncovered & neighbor_bitsets[c]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            
            if not candidates:
                break
                
            c = rng.choice(candidates)
            chosen.append(c)
            uncovered &= ~neighbor_bitsets[c]
            
            if uncovered == 0:
                break
                
        return chosen

    def _simulated_annealing(initial_code, seed):
        """
        Simulated annealing to minimize code size.
        Moves:
        1. Remove a random codeword (if valid).
        2. Swap a random codeword with a random other word (if valid and allows future reduction? 
           Actually, simple swap is only useful if it leads to a state where removal is possible.
           Better: 
           - Try to remove a random codeword. If it fails, it's a "blocked" state.
           - To escape, perform a "swap": replace one codeword with another. 
           The new codeword should ideally cover some of the same ground, or allow another to be removed.
           Since we want to minimize size, we don't want to just swap. We want to swap to enable a removal.
           
           Strategy:
           - State: list of codewords.
           - Energy: len(code). We want to minimize.
           - Move 1: Pick random i. Try to remove code[i]. If valid, accept (energy decreases).
           - Move 2: Pick random i. Pick random j (j not in code). Swap code[i] with j.
             Check if new code is valid. 
             If valid, we have same size. Why do it? 
             To change the structure so that a future removal might work.
             This is a neutral move. In SA, we accept neutral moves with some probability to explore.
             But simply swapping might not help if we don't try to remove immediately after.
             
           Improved Move 2: "Perturbation".
             Pick random i. 
             Generate a few random candidates j.
             For each j, form new_code = code with code[i] replaced by j.
             If new_code is valid, check if any word in new_code is now removable.
             If yes, accept the swap AND the removal? That's a 2-step move.
             Or just accept the swap if it reduces the "blockedness"?
             
           Let's stick to a simpler heuristic search:
           1. Greedy removal.
           2. If stuck, try random swaps.
              - Pick i. Pick random j not in code.
              - new_code = code - {code[i]} + {j}.
              - If new_code is valid:
                  - Try greedy removal on new_code.
                  - If size decreases, accept.
                  - Else, with probability p, accept the swap (to explore).
        """
        rng = random.Random(seed)
        code = initial_code[:]
        code = _remove_redundant(code)
        
        current_size = len(code)
        best_code = code[:]
        best_size = current_size
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        # Precompute all indices for random selection
        all_indices = list(range(total))
        
        while _time_ok() and T > T_min:
            T *= cooling
            
            # Try to remove a random element
            if len(code) > 0:
                i = rng.randrange(len(code))
                c_to_remove = code[i]
                
                # Check if removable
                others_union = 0
                for j in range(len(code)):
                    if j != i:
                        others_union |= neighbor_bitsets[code[j]]
                
                if (neighbor_bitsets[c_to_remove] & ~others_union) == 0:
                    # Valid removal
                    code.pop(i)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code = code[:]
                    # Continue to next iteration to try more removals
                    continue
            
            # If no removal possible (or we chose to explore), try a swap
            if len(code) == 0:
                break
                
            i = rng.randrange(len(code))
            c_old = code[i]
            
            # Pick a random new codeword
            # To make swaps more effective, maybe pick one that covers some unique bits of c_old?
            # Or just random.
            # Random might be slow. Let's pick from a subset of "promising" candidates?
            # For now, random from all_indices.
            c_new = rng.choice(all_indices)
            
            if c_new == c_old:
                continue
                
            # Check if c_new is already in code
            if c_new in code:
                continue
                
            # Form new code
            new_code = code[:]
            new_code[i] = c_new
            
            # Check validity
            union = 0
            for c in new_code:
                union |= neighbor_bitsets[c]
            if union != all_bits:
                continue
            
            # New code is valid. Same size.
            # Try to remove redundant elements from new_code
            new_code_reduced = _remove_redundant(new_code)
            
            if len(new_code_reduced) < current_size:
                # Improvement!
                code = new_code_reduced
                current_size = len(code)
                if current_size < best_size:
                    best_size = current_size
                    best_code = code[:]
            else:
                # No immediate improvement. Accept swap with probability based on T?
                # Since size didn't change, it's a neutral move.
                # In standard SA, we accept neutral moves to explore.
                # Let's accept with probability 0.5 to mix up the state.
                if rng.random() < 0.5:
                    code = new_code # Keep the swap, even if no reduction happened
                    # Note: We don't update best_code here because size didn't improve.
        
        return best_code

    best_code = None
    best_len = total + 1
    
    # Phase 1: Greedy to get a feasible start
    # Do a few greedy restarts to find a good starting point
    for i in range(5):
        if not _time_ok():
            break
        code = _randomized_greedy(i)
        code = _remove_redundant(code)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    if best_code is None:
        # Fallback
        best_code = _randomized_greedy(0)
        best_code = _remove_redundant(best_code)
        best_len = len(best_code)
        
    # Phase 2: Simulated Annealing / Local Search
    # Run multiple SA instances starting from the best greedy solution or random restarts
    num_sa_restarts = 20
    for i in range(num_sa_restarts):
        if not _time_ok():
            break
        
        # Start from best known or a fresh greedy?
        # Starting from best known is good for polishing.
        # Starting from fresh greedy is good for exploration.
        # Let's alternate.
        if i % 2 == 0:
            start_code = best_code[:]
        else:
            start_code = _randomized_greedy(100 + i)
            start_code = _remove_redundant(start_code)
            if len(start_code) < best_len:
                best_len = len(start_code)
                best_code = start_code[:]
        
        sa_result = _simulated_annealing(start_code, 1000 + i)
        if len(sa_result) < best_len:
            best_len = len(sa_result)
            best_code = sa_result[:]
            
    # Final polish
    best_code = _remove_redundant(best_code)
    
    # Final validation and conversion
    covered = 0
    for c in best_code:
        covered |= neighbor_bitsets[c]
    if covered != all_bits:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    out = []
    for c in best_code:
        out.append(list(np.unravel_index(c, (q,) * n)))
    return out
