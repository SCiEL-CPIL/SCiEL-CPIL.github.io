import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    # Greedy initialization
    def greedy():
        values = []
        sums = set()
        for x in range(1, N + 1):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    init_seq, init_sums = greedy()
    best_seq = init_seq
    best_len = len(init_seq)
    
    # DFS with pruning - optimized for speed
    def dfs(start_val, seq, suffix_sums, sums_set):
        nonlocal best_seq, best_len
        
        # Pruning: if we can't beat the best length, return
        if len(seq) + (N - start_val + 1) <= best_len:
            return
        
        # Time check
        if time.time() > deadline:
            return
        
        # Iterate candidates
        for x in range(start_val, N + 1):
            # Calculate new suffix sums
            new_suffix_sums = [x]
            valid = True
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                if new_suffix_sums[-1] in sums_set:
                    valid = False
                    break
            
            if x in sums_set:
                valid = False
            
            if valid:
                new_sums_set = sums_set.union(new_suffix_sums)
                new_seq = seq + [x]
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
                
                # Recurse
                dfs(x + 1, new_seq, new_suffix_sums, new_sums_set)

    # Run DFS to find a longer sequence
    dfs(1, [], [], set())
    
    # Local search: try to improve by single-element replacements and re-extension
    def _block_sums(seq):
        """Return set of all nonempty consecutive block sums."""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def _valid(seq):
        """Check if seq has all distinct consecutive block sums and is strictly increasing in [1,256]."""
        if not seq:
            return True
        if any(x < 1 or x > 256 for x in seq):
            return False
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        return len(_block_sums(seq)) == len(seq) * (len(seq) + 1) // 2

    def try_extend(seq, sums):
        """Try to greedily extend seq. Return (new_seq, new_sums, extended)."""
        new_seq = list(seq)
        new_sums = set(sums)
        extended = False
        for x in range(new_seq[-1] + 1, N + 1) if new_seq else range(1, N + 1):
            tails = []
            total = 0
            for a in reversed(new_seq + [x]):
                total += a
                tails.append(total)
            tail_set = set(tails)
            if not new_sums.intersection(tail_set):
                new_seq.append(x)
                new_sums.update(tail_set)
                extended = True
        return new_seq, new_sums, extended

    def single_replacement_search(seq, sums, time_limit):
        """Try single-element replacements and re-extend."""
        best_local = (list(seq), set(sums))
        n = len(seq)
        
        for i in range(n):
            if time.time() > time_limit:
                break
            old = seq[i]
            lo = seq[i-1] + 1 if i > 0 else 1
            hi = seq[i+1] - 1 if i < n-1 else N
            # Try a few candidate replacements
            candidates = set()
            # Try nearby values
            for dx in [-3, -2, -1, 1, 2, 3]:
                v = old + dx
                if lo <= v <= hi:
                    candidates.add(v)
            # Try random values
            for _ in range(10):
                if lo < hi:
                    candidates.add(random.randint(lo, hi))
            
            for v in candidates:
                if v == old:
                    continue
                new_seq = list(seq)
                new_seq[i] = v
                # Check strictly increasing
                valid_incr = True
                for k in range(1, len(new_seq)):
                    if new_seq[k] <= new_seq[k-1]:
                        valid_incr = False
                        break
                if not valid_incr:
                    continue
                # Check distinct block sums
                new_sums = _block_sums(new_seq)
                if len(new_sums) != len(new_seq) * (len(new_seq) + 1) // 2:
                    continue
                # Try to extend
                ext_seq, ext_sums, _ = try_extend(new_seq, new_sums)
                if len(ext_seq) > len(best_local[0]):
                    best_local = (ext_seq, ext_sums)
                    if len(ext_seq) > best_len:
                        best_len = len(ext_seq)
                        best_seq = ext_seq[:]
        return best_local

    # Iterate local search
    improved = True
    random.seed(42)
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to extend
        ext_seq, ext_sums, ext = try_extend(best_seq, best_sums if 'best_sums' in dir() else _block_sums(best_seq))
        if ext and len(ext_seq) > best_len:
            best_seq, best_sums = ext_seq, ext_sums
            best_len = len(ext_seq)
            improved = True
            continue
        # Try single replacement
        new_seq, new_sums = single_replacement_search(best_seq, _block_sums(best_seq), deadline - 5)
        if len(new_seq) > best_len:
            best_seq, best_sums = new_seq, new_sums
            best_len = len(new_seq)
            improved = True
    
    return best_seq
