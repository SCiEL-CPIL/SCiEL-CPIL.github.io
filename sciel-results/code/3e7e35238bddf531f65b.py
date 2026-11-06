<<<<<<< SEARCH
def _swap_improve(seq, processing_times, max_iter=200):
    """Iterated swap local search. Returns improved seq and makespan."""
    n = len(seq)
    if n <= 1:
        return seq, _makespan(seq, processing_times)
    cur = np.array(seq)
    cur_ms = _makespan(cur, processing_times)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # swap i and j
                cur[i], cur[j] = cur[j], cur[i]
                ms = _makespan(cur, processing_times)
                if ms < cur_ms - 1e-9:
                    cur_ms = ms
                    improved = True
                else:
                    cur[i], cur[j] = cur[j], cur[i]
    return cur.tolist(), cur_ms


def _or_improve(seq, processing_times, max_iter=100):
    """Or-opt: move a block of 1-3 jobs to another position."""
    n = len(seq)
    if n <= 2:
        return seq, _makespan(seq, processing_times)
    cur = list(seq)
    cur_ms = _makespan(cur, processing_times)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for block_len in (1, 2, 3):
            for i in range(n - block_len + 1):
                block = cur[i:i + block_len]
                # remove block
                rest = cur[:i] + cur[i + block_len:]
                for pos in range(len(rest) + 1):
                    if pos == i:
                        continue
                    cand = rest[:pos] + block + rest[pos:]
                    ms = _makespan(cand, processing_times)
                    if ms < cur_ms - 1e-9:
                        cur = cand
                        cur_ms = ms
                        improved = True
                        # restart block search from top for this block_len
                        break
                if improved:
                    break
            if improved:
                break
    return cur, cur_ms


def _random_seq(n, rng):
    return rng.permutation(n).tolist()


def _perturb(seq, processing_times, rng, strength=3):
    """Randomly swap/insert a few jobs to escape local minima."""
    n = len(seq)
    cur = list(seq)
    for _ in range(strength):
        op = rng.random()
        if op < 0.5:
            # swap
            i, j = rng.sample(range(n), 2)
            cur[i], cur[j] = cur[j], cur[i]
        else:
            # or-opt move
            block_len = rng.choice([1, 2, 3])
            if n < block_len:
                continue
            i = rng.randrange(n - block_len + 1)
            block = cur[i:i + block_len]
            rest = cur[:i] + cur[i + block_len:]
            pos = rng.randrange(len(rest) + 1)
            if pos > i:
                pos -= 0  # index in rest
            cur = rest[:pos] + block + rest[pos:]
    return cur
=======
def _tabu_search(seq, processing_times, time_limit, rng):
    """
    Tabu Search for Flow Shop Problem.
    Moves: Swap two jobs (i, j).
    Tabu List: Stores (i, j) pairs that are forbidden for a certain number of iterations.
    """
    n = len(seq)
    if n <= 1:
        return seq, _makespan(seq, processing_times)
    
    cur_seq = np.array(seq, dtype=int)
    cur_ms = _makespan(cur_seq, processing_times)
    best_seq = cur_seq.copy()
    best_ms = cur_ms
    
    # Tabu list: dictionary mapping (min(i,j), max(i,j)) to the iteration when it becomes non-tabu
    tabu_list = {}
    tabu_tenure = 10 # Dynamic tenure could be better, but fixed is okay for start
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        best_move = None
        best_move_ms = np.inf
        best_move_i = -1
        best_move_j = -1
        best_is_tabu = False
        best_tabu_release = 0
        
        # Generate neighbor moves: swaps
        for i in range(n):
            for j in range(i + 1, n):
                # Check tabu
                key = (i, j)
                is_tabu = key in tabu_list and tabu_list[key] > 0
                
                # Apply move
                cur_seq[i], cur_seq[j] = cur_seq[j], cur_seq[i]
                ms = _makespan(cur_seq, processing_times)
                # Undo move
                cur_seq[i], cur_seq[j] = cur_seq[j], cur_seq[i]
                
                # Aspiration criterion: accept if it improves best known
                if ms < best_ms - 1e-9:
                    best_is_tabu = False
                elif is_tabu:
                    # If tabu and not improving best, skip or check release time
                    # We will only consider it if it's the best available and we have no non-tabu options?
                    # Standard TS: choose best non-tabu move. If all are tabu, choose best tabu move.
                    # For simplicity, we track best non-tabu and best overall.
                    pass
                else:
                    best_is_tabu = False

                # Logic: 
                # 1. Find best non-tabu move.
                # 2. If no non-tabu move, find best tabu move.
                
                # Let's accumulate candidates
                if not is_tabu or ms < best_ms - 1e-9:
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move_i = i
                        best_move_j = j
                        best_is_tabu = is_tabu
                        best_tabu_release = tabu_list.get(key, 0)
                        
        if best_move_i == -1:
            # No moves found? (Shouldn't happen unless n=1)
            break
            
        # Apply the best move
        cur_seq[best_move_i], cur_seq[best_move_j] = cur_seq[best_move_j], cur_seq[best_move_i]
        cur_ms = best_move_ms
        
        # Update tabu list
        # We need to decrement counters or just store expiration
        # For efficiency, we can just store the iteration number when it expires
        # But we don't have a global iteration counter easily accessible in this loop structure without passing it.
        # Let's use a simple list of (i, j, expiration_time)
        # To keep it simple and fast, let's use a set of (i, j) that are tabu, and a queue to remove them.
        
        # Simpler Tabu implementation:
        # Keep a list of recent moves. If a move is in the list, it's tabu.
        # Move list size = tabu_tenure.
        
        # Let's refactor to use a move history list
        pass # This placeholder indicates the logic above was a bit messy. I will rewrite the function body in the next block.
        
    return best_seq.tolist(), best_ms
>>>>>>> REPLACE

