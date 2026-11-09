<<<<<<< SEARCH
import time

def construct():
    start = time.time()
    DEADLINE = start + 80

    def build_greedy_smallest():
        values = []
        sums = set()
        n = 0
        for x in range(1, 257):
            if time.time() > DEADLINE - 1:
                break
            # New consecutive sums ending at x:
            # x, x + values[-1], x + values[-1] + values[-2], ...
            t = 0
            ok = True
            for a in reversed(values):
                t += a
                if t in sums:
                    ok = False
                    break
            if not ok:
                continue
            t = x
            if t in sums:
                continue
            # All good, add x
            new_sums = [x]
            t = x
            for a in reversed(values):
                t += a
                new_sums.append(t)
            values.append(x)
            sums.update(new_sums)
            n += 1
        return values

    def build_greedy_largest():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            if time.time() > DEADLINE - 1:
                break
            t = 0
            ok = True
            for a in reversed(values):
                t += a
                if t in sums:
                    ok = False
                    break
            if not ok:
                continue
            t = x
            if t in sums:
                continue
            new_sums = [x]
            t = x
            for a in reversed(values):
                t += a
                new_sums.append(t)
            values.append(x)
            sums.update(new_sums)
        return sorted(values)

    def build_local_search():
        """Start from greedy_smallest, then try local modifications."""
        seq = build_greedy_smallest()
        if not seq:
            return seq

        def compute_sums(s):
            out = set()
            n = len(s)
            for i in range(n):
                acc = 0
                for j in range(i, n):
                    acc += s[j]
                    out.add(acc)
            return out

        def valid_insert(s, pos, val):
            """Check if inserting val at position pos keeps all sums distinct."""
            n = len(s)
            # New sums involving val:
            # All sums that now pass through val's position.
            # Sums of [s[i..pos-1]] + val, [s[i..pos-1]] + val + [s[pos..j]]
            # Actually, let's just compute all sums of the new sequence
            new_s = s[:pos] + [val] + s[pos:]
            if not (1 <= val <= 256):
                return False, None
            # Check strictly increasing
            for k in range(len(new_s)-1):
                if new_s[k] >= new_s[k+1]:
                    return False, None
            sset = compute_sums(new_s)
            if len(sset) != len(new_s)*(len(new_s)+1)//2:
                return False, None
            return True, new_s

        def valid_delete(s, pos):
            new_s = s[:pos] + s[pos+1:]
            if not new_s:
                return new_s
            sset = compute_sums(new_s)
            if len(sset) == len(new_s)*(len(new_s)+1)//2:
                return new_s
            return None

        best = seq
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            current = best
            n = len(current)

            # Try inserting each value 1..256 at each position
            for x in range(1, 257):
                if time.time() > DEADLINE - 5:
                    break
                if x in current:
                    continue
                for pos in range(n+1):
                    ok, new_s = valid_insert(current, pos, x)
                    if ok and len(new_s) > len(best):
                        best = new_s
                        improved = True

            # If we got longer, also try single swaps to fix
            if improved:
                # After insertion, try to remove any redundant elements to free space
                n = len(best)
                for pos in range(n):
                    if time.time() > DEADLINE - 5:
                        break
                    new_s = valid_delete(best, pos)
                    if new_s and len(new_s) > len(best) - 1:
                        # This deletion doesn't help length, but might enable future insertions
                        # Only keep if it doesn't reduce length
                        pass

        return best

    # Try all strategies, keep best
    candidates = []

    s1 = build_greedy_smallest()
    candidates.append(s1)

    if time.time() < DEADLINE - 20:
        s2 = build_greedy_largest()
        candidates.append(s2)

    if time.time() < DEADLINE - 10:
        s3 = build_local_search()
        candidates.append(s3)

    best = max(candidates, key=len)

    # Verify the best result
    n = len(best)
    if n == 0:
        return []
    # Check strictly increasing
    for i in range(n-1):
        if best[i] >= best[i+1]:
            return best[:i+1]
    # Check all values in [1,256]
    for v in best:
        if v < 1 or v > 256:
            return []
    # Check distinct consecutive sums
    sset = set()
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += best[j]
            if acc in sset:
                return best[:j]  # trim to valid prefix
            sset.add(acc)

    return best
=======
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 80
    
    # Precompute all possible consecutive block sums for a given sequence
    def get_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += seq[j]
                sums.add(acc)
        return sums
    
    # Check if a sequence is valid (strictly increasing, distinct block sums)
    def is_valid(seq):
        if not seq:
            return True
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        # Check all values in [1, 256]
        for v in seq:
            if v < 1 or v > 256:
                return False
        # Check distinct consecutive sums
        sums = get_block_sums(seq)
        expected = len(seq) * (len(seq) + 1) // 2
        return len(sums) == expected
    
    # Greedy construction: try to extend a sequence by adding one element
    def try_extend(seq, candidate_vals):
        """Try to add each candidate value to seq, return best extension"""
        best = None
        best_len = len(seq)
        
        for pos in range(len(seq) + 1):
            for x in candidate_vals:
                # Check if x can be inserted at pos maintaining strictly increasing
                if pos > 0 and x <= seq[pos - 1]:
                    continue
                if pos < len(seq) and x >= seq[pos]:
                    continue
                if x < 1 or x > 256:
                    continue
                
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    if len(new_seq) > best_len or (best is None):
                        best = new_seq
                        best_len = len(new_seq)
        
        return best
    
    # Build initial sequence using greedy approach
    def build_greedy(direction='smallest'):
        seq = []
        if direction == 'smallest':
            candidates = list(range(1, 257))
        else:
            candidates = list(range(256, 0, -1))
        
        while time.time() < DEADLINE - 30:
            extended = try_extend(seq, candidates)
            if extended is None or len(extended) == len(seq):
                break
            seq = extended
            # Remove used values from candidates
            used = set(seq)
            candidates = [x for x in candidates if x not in used]
        
        return seq
    
    # Local search: try to improve by swapping elements
    def local_search(seq):
        if not seq:
            return seq
        
        best = list(seq)
        improved = True
        random.seed(42)
        
        while improved and time.time() < DEADLINE - 10:
            improved = False
            current = list(best)
            n = len(current)
            
            # Try single element replacement
            for pos in range(n):
                if time.time() > DEADLINE - 10:
                    break
                old_val = current[pos]
                # Try replacing with nearby values
                for delta in range(1, 50):
                    for new_val in [old_val - delta, old_val + delta]:
                        if new_val < 1 or new_val > 256:
                            continue
                        if new_val in current:
                            continue
                        
                        # Check if replacement maintains strictly increasing
                        if pos > 0 and new_val <= current[pos - 1]:
                            continue
                        if pos < n - 1 and new_val >= current[pos + 1]:
                            continue
                        
                        new_seq = current.copy()
                        new_seq[pos] = new_val
                        if is_valid(new_seq):
                            # Accept if it allows future improvements
                            # For now, just accept valid replacements
                            if random.random() < 0.3:  # Random acceptance
                                best = new_seq
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try inserting a new element
            if not improved:
                for pos in range(len(best) + 1):
                    if time.time() > DEADLINE - 10:
                        break
                    for x in range(1, 257):
                        if x in best:
                            continue
                        if pos > 0 and x <= best[pos - 1]:
                            continue
                        if pos < len(best) and x >= best[pos]:
                            continue
                        
                        new_seq = best[:pos] + [x] + best[pos:]
                        if is_valid(new_seq):
                            best = new_seq
                            improved = True
                            break
                    if improved:
                        break
            
            # Try deleting an element to free up space
            if not improved and len(best) > 5:
                for pos in range(len(best)):
                    if time.time() > DEADLINE - 10:
                        break
                    new_seq = best[:pos] + best[pos + 1:]
                    if is_valid(new_seq):
                        # Only delete if it might enable better insertions
                        if random.random() < 0.1:
                            best = new_seq
                            improved = True
                            break
        
        return best
    
    # Try multiple strategies
    candidates = []
    
    # Strategy 1: Greedy smallest
    s1 = build_greedy('smallest')
    candidates.append(s1)
    
    # Strategy 2: Greedy largest
    if time.time() < DEADLINE - 25:
        s2 = build_greedy('largest')
        candidates.append(s2)
    
    # Strategy 3: Local search from greedy smallest
    if time.time() < DEADLINE - 15:
        s3 = local_search(build_greedy('smallest'))
        candidates.append(s3)
    
    # Strategy 4: Local search from greedy largest
    if time.time() < DEADLINE - 10:
        s4 = local_search(build_greedy('largest'))
        candidates.append(s4)
    
    # Strategy 5: Randomized restarts
    if time.time() < DEADLINE - 5:
        random.seed(123)
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            # Start with a random subset
            initial = random.sample(range(1, 257), min(10, 256))
            initial.sort()
            if not is_valid(initial):
                # Keep only valid prefix
                while initial and not is_valid(initial):
                    initial.pop()
            
            extended = local_search(initial)
            candidates.append(extended)
    
    # Find the best candidate
    best = max(candidates, key=len)
    
    # Final verification
    if not is_valid(best):
        # Trim to valid prefix
        while best and not is_valid(best):
            best.pop()
    
    return best
>>>>>>> REPLACE
