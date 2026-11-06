import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    best_seq = []
    best_len = 0
    
    def try_greedy(start_offset=1, step=1, max_val=256):
        """
        Build a sequence using a greedy approach with a twist:
        Instead of always picking the smallest valid number, we can start from different offsets
        and use different step patterns to explore different regions of the value space.
        """
        seq = []
        prefixes = [0]
        diffs = set()
        last_a = 0
        
        while time.time() - start_time < deadline:
            found = False
            # Try candidates starting from last_a + start_offset, then + start_offset + step, etc.
            # This creates different "patterns" of selection
            for a in range(last_a + start_offset, max_val + 1, 1):
                new_p = prefixes[-1] + a
                # Check for collisions with all existing block sums
                ok = True
                for p in prefixes:
                    d = new_p - p
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    # Add new differences
                    for p in prefixes:
                        diffs.add(new_p - p)
                    prefixes.append(new_p)
                    seq.append(a)
                    last_a = a
                    found = True
                    break
            
            if not found:
                break
        
        return seq
    
    def try_random_window(window_sizes=None):
        """
        Build a sequence by randomly selecting from a window of candidates.
        Different window sizes lead to different exploration patterns.
        """
        if window_sizes is None:
            window_sizes = [10, 20, 50, 100, 200]
        
        seq = []
        prefixes = [0]
        diffs = set()
        last_a = 0
        
        while time.time() - start_time < deadline:
            found = False
            w = random.choice(window_sizes)
            upper = min(256, last_a + w)
            if upper <= last_a:
                break
            
            candidates = list(range(last_a + 1, upper + 1))
            random.shuffle(candidates)
            
            for a in candidates:
                new_p = prefixes[-1] + a
                ok = True
                for p in prefixes:
                    d = new_p - p
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    for p in prefixes:
                        diffs.add(new_p - p)
                    prefixes.append(new_p)
                    seq.append(a)
                    last_a = a
                    found = True
                    break
            
            if not found:
                break
        
        return seq
    
    def try_local_search(seq):
        """
        Given a valid sequence, try to improve it by swapping or replacing elements.
        This is a structural change: we're not just building from scratch, but refining.
        """
        if not seq:
            return seq
        
        best_local = list(seq)
        improved = True
        
        while improved and time.time() - start_time < deadline:
            improved = False
            current = list(best_local)
            
            # Try replacing each element with a nearby valid one
            for i in range(len(current)):
                orig = current[i]
                # Try values in a neighborhood around orig
                for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]:
                    new_val = orig + delta
                    if new_val < 1 or new_val > 256:
                        continue
                    if i > 0 and new_val <= current[i-1]:
                        continue
                    if i < len(current) - 1 and new_val >= current[i+1]:
                        continue
                    
                    # Check if the new sequence is valid
                    test_seq = current[:i] + [new_val] + current[i+1:]
                    if _is_valid(test_seq):
                        best_local = test_seq
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_local
    
    def _is_valid(seq):
        """Check if a sequence has all distinct consecutive block sums."""
        if not seq:
            return True
        if len(seq) > 1 and seq[0] >= seq[1]:
            return False
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        
        # Check all consecutive block sums
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Strategy 1: Deterministic greedy with different starting offsets
    for offset in [1, 2, 3, 5, 8]:
        if time.time() - start_time >= deadline:
            break
        s = try_greedy(start_offset=offset)
        if len(s) > best_len:
            best_len = len(s)
            best_seq = s
    
    # Strategy 2: Randomized restarts with various window sizes
    while time.time() - start_time < deadline - 5:
        s = try_random_window()
        if len(s) > best_len:
            best_len = len(s)
            best_seq = s
    
    # Strategy 3: Local search refinement on the best sequence found
    if best_seq and time.time() - start_time < deadline - 2:
        refined = try_local_search(best_seq)
        if len(refined) > best_len:
            best_len = len(refined)
            best_seq = refined
    
    return best_seq

