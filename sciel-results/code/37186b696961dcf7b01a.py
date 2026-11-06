import time
import random

def construct():
    """
    Returns a strictly increasing sequence of integers in [1, 256] with distinct consecutive-block sums.
    Uses multi-start randomized construction with variable window sizes.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0

    def try_build(window_size=None, greedy_mode=False, seed=None):
        """
        Builds a sequence using a randomized or greedy strategy.
        window_size: If not greedy, pick next element from [last+1, last+window_size].
                     If greedy, pick smallest valid element.
        """
        if seed is not None:
            random.seed(seed)
            
        seq = []
        # Prefix sums P[0]=0, P[i] = sum(seq[0..i-1])
        # Block sums are P[j] - P[i] for i < j.
        # We maintain a set of all existing block sums.
        diffs = set()
        prefixes = [0]
        last_a = 0
        
        # Optional: Start with a specific value? 
        # The prompt implies we can start anywhere. 
        # Starting with 1 is usually best for length, but random starts help diversity.
        
        while True:
            if time.time() > deadline:
                return seq
            
            n = len(seq)
            # Determine candidate range
            if greedy_mode:
                candidates = range(last_a + 1, 257)
            else:
                if window_size is None:
                    window_size = random.choice([10, 20, 50, 100, 150, 200])
                upper = min(256, last_a + window_size)
                if upper <= last_a:
                    break
                candidates = list(range(last_a + 1, upper + 1))
                random.shuffle(candidates)
            
            found = False
            for a in candidates:
                # Check if adding 'a' creates a conflict
                # New prefix: P_new = prefixes[-1] + a
                # New diffs: P_new - P_i for all i in 0..n
                # Note: prefixes has length n+1 (indices 0 to n)
                
                p_new = prefixes[-1] + a
                
                conflict = False
                for i in range(n + 1):
                    d = p_new - prefixes[i]
                    if d in diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept 'a'
                    seq.append(a)
                    prefixes.append(p_new)
                    # Update diffs
                    for i in range(n + 1):
                        diffs.add(p_new - prefixes[i])
                    last_a = a
                    found = True
                    break
            
            if not found:
                break
                
        return seq

    # 1. Deterministic Greedy (Smallest Valid)
    # This provides a strong baseline.
    s = try_build(greedy_mode=True)
    if len(s) > best_len:
        best_len = len(s)
        best_seq = s[:]

    # 2. Randomized Restarts
    # We iterate until deadline.
    # We vary the window size to explore different "densities" of the sequence.
    # Small windows: dense, short.
    # Large windows: sparse, potentially longer if we avoid conflicts.
    
    iteration = 0
    while time.time() < deadline - 2.0:
        iteration += 1
        
        # Strategy: Mix of small and large windows
        # Also, occasionally try to start with a specific small number to force density
        # but pick random subsequent numbers.
        
        # Random window size
        # Bias towards larger windows sometimes to escape local traps
        r = random.random()
        if r < 0.3:
            w = random.randint(5, 20)
        elif r < 0.7:
            w = random.randint(20, 60)
        else:
            w = random.randint(60, 200)
            
        s = try_build(window_size=w)
        
        if len(s) > best_len:
            best_len = len(s)
            best_seq = s[:]
            
        # If we are close to the deadline, we might want to try a "repair" 
        # or a greedy extension on the best found so far.
        # But try_build already does a full run.
        
    # Final check: Try to extend best_seq greedily one more time?
    # The try_build function stops when no valid move is found in the window.
    # If greedy_mode was False, it might have stopped early because the window ran out.
    # We can try to extend the best_seq found so far with a greedy pass.
    
    if best_seq and time.time() < deadline - 1.0:
        # Create a state from best_seq
        # We can just call try_build with the sequence pre-filled? 
        # My try_build starts from empty. 
        # Let's write a helper to extend an existing sequence.
        
        def extend_sequence(seq):
            if not seq:
                return seq
            # Rebuild state
            diffs = set()
            prefixes = [0]
            for a in seq:
                p_new = prefixes[-1] + a
                prefixes.append(p_new)
                for i in range(len(prefixes) - 1):
                    diffs.add(p_new - prefixes[i])
            
            last_a = seq[-1]
            # Try to extend with greedy smallest
            while True:
                if time.time() > deadline:
                    break
                n = len(seq)
                found = False
                for a in range(last_a + 1, 257):
                    p_new = prefixes[-1] + a
                    conflict = False
                    for i in range(n + 1):
                        d = p_new - prefixes[i]
                        if d in diffs:
                            conflict = True
                            break
                    if not conflict:
                        seq.append(a)
                        prefixes.append(p_new)
                        for i in range(n + 1):
                            diffs.add(p_new - prefixes[i])
                        last_a = a
                        found = True
                        break
                if not found:
                    break
            return seq

        extended = extend_sequence(best_seq[:])
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(extended)

    # Verify validity of the best sequence found
    # Just in case of bugs
    def _check_valid(seq):
        if not seq:
            return True
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        if any(x < 1 or x > 256 for x in seq):
            return False
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        expected = n * (n + 1) // 2
        if len(sums) != expected:
            return False
        return True

    if not _check_valid(best_seq):
        # Fallback to a known small valid sequence
        # [1, 2, 5] -> sums: 1, 2, 5, 3, 7, 8. All distinct.
        # [1, 2, 5, 10]? 
        # 1, 2, 5, 10
        # 1, 2, 5, 10
        # 3, 7, 15
        # 12, 17
        # 22
        # Sums: 1,2,5,10,3,7,15,12,17,22. All distinct.
        # Let's use a safe fallback
        best_seq = [1, 2, 5, 10]
        
    return best_seq
