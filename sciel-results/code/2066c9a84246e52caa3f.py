"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def _compute_tails(values, x):
    """Compute all new tail sums when appending x to values."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _is_valid_extension(values, sums, x):
    """Check if appending x creates any duplicate block sums."""
    tails = _compute_tails(values, x)
    return not sums.intersection(tails)

def _extend_greedy(values, sums, candidates, deadline):
    """Extend values greedily using smallest valid candidate."""
    for x in candidates:
        if time.time() > deadline:
            break
        if _is_valid_extension(values, sums, x):
            tails = _compute_tails(values, x)
            values.append(x)
            sums.update(tails)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Strategy 1: Greedy smallest
    def greedy_smallest():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 2: Greedy largest
    def greedy_largest():
        values, sums = [], set()
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return sorted(values)
    
    # Strategy 3: Beam search with multiple partial sequences
    def beam_search():
        """
        Beam search maintaining top-K partial sequences.
        Each beam entry: (neg_length, last_val, total_sum, values, sums)
        We prioritize sequences that can be extended further.
        """
        beam_width = 10
        # Each entry: (priority, last_val, total, values, sums)
        # Priority is negative length (so we pop longest first), then last_val (prefer smaller last_val for more room)
        beam = [(0, 0, 0, [], set())]  # (neg_len, last_val, total, values, sums)
        
        best_result = []
        
        while beam and time.time() < deadline - 2:
            # Expand all beams
            new_beam = []
            for neg_len, last_val, total, values, sums in beam:
                # Try extending this sequence
                for x in range(last_val + 1, 257):
                    if time.time() > deadline - 2:
                        break
                    if _is_valid_extension(values, sums, x):
                        tails = _compute_tails(values, x)
                        new_values = values + [x]
                        new_sums = sums | set(tails)
                        new_total = total + x
                        # Priority: longer is better, then smaller last value (more room to grow)
                        new_neg_len = -(len(new_values))
                        new_last = x
                        new_beam.append((new_neg_len, new_last, new_total, new_values, new_sums))
                
                # Also keep current state as a candidate (in case we can't extend)
                new_beam.append((neg_len, last_val, total, values, sums))
            
            if not new_beam:
                break
            
            # Sort and keep top beam_width
            new_beam.sort(key=lambda e: (e[0], e[1]))  # Sort by neg_len (so most negative = longest first), then last_val
            beam = new_beam[:beam_width]
            
            # Track best
            for entry in beam:
                if len(entry[3]) > len(best_result):
                    best_result = entry[3]
        
        return best_result
    
    # Strategy 4: Iterative deepening with pruning
    def iterative_deepening():
        """Try to build sequences of increasing length."""
        best_result = []
        
        # First get a baseline from greedy
        baseline = greedy_smallest()
        best_result = baseline
        
        target_len = len(baseline) + 1
        
        while target_len <= 20 and time.time() < deadline - 10:
            # Try to find a sequence of length target_len
            found = _find_sequence_of_length(target_len, deadline - 5)
            if found:
                best_result = found
                target_len += 1
            else:
                break
        
        return best_result
    
    def _find_sequence_of_length(target_len, time_budget):
        """Try to find a strictly increasing sequence of exactly target_len with distinct block sums."""
        # Use backtracking with pruning
        def backtrack(values, sums, min_val, depth):
            if time.time() > time_budget:
                return None
            if len(values) == target_len:
                return values[:]
            
            # Pruning: even if we pick the smallest possible values, can we reach target_len?
            # Max remaining slots = 256 - min_val
            if 256 - min_val + 1 < target_len - len(values):
                return None
            
            # Try candidates from min_val to 256
            for x in range(min_val, 257):
                if time.time() > time_budget:
                    return None
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    new_sums = sums | set(tails)
                    result = backtrack(values + [x], new_sums, x + 1, depth + 1)
                    if result is not None:
                        return result
            return None
        
        return backtrack([], set(), 1, 0)
    
    # Run all strategies, keep the best
    best_result = []
    
    strategies = [greedy_smallest, greedy_largest, beam_search, iterative_deepening]
    
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        try:
            result = strategy()
            if len(result) > len(best_result):
                best_result = result
        except Exception:
            pass
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
