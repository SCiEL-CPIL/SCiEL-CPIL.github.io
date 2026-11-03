import time
from collections import Counter
import itertools

def construct(n):
    # 1. Generate classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if n <= 6:
        return "".join(map(str, word))

    # 2. Optimization for n=7 (and potentially larger, though time constrained)
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Initialize Counter
    L = len(word)
    perm_counts = Counter()
    for i in range(L - n + 1):
        perm_counts[tuple(word[i:i+n])] += 1
        
    # All n! permutations should be present
    # We track how many permutations have count 0
    missing_count = 0
    for p in itertools.permutations(range(1, n+1)):
        if perm_counts[p] == 0:
            missing_count += 1
            
    # If not all present, something is wrong with classic construction? 
    # Classic construction is valid, so missing_count should be 0.
    
    # Greedy removal
    # We try to remove characters one by one.
    # To handle index shifting, we can iterate from end to start?
    # Or just restart the scan if a removal is successful.
    # Given L ~ 6000, and we expect ~8 removals, restarting is fine.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Scan from left to right
        for i in range(len(word)):
            if time.time() > deadline:
                break
            
            # Calculate windows to lose
            lost_windows = []
            # Windows in word[i-n+1 : i+1] to word[i : i+n]
            # Start indices: max(0, i-n+1) to min(i, len(word)-n)
            start_min = max(0, i - n + 1)
            start_max = min(i, len(word) - n)
            
            for j in range(start_min, start_max + 1):
                lost_windows.append(tuple(word[j:j+n]))
                
            # Calculate windows to gain
            # These are the windows crossing the junction in word'
            # word' = word[:i] + word[i+1:]
            # Junction is between word[i-1] and word[i+1]
            # Windows are word[i-k : i] + word[i+1 : i+1 + (n-k)] for k=1..n-1
            # Wait, k is length of left part.
            # Left part length k, right part length n-k.
            # k ranges from 1 to n-1.
            # Also need to check bounds.
            gained_windows = []
            for k in range(1, n):
                left_start = i - k
                right_end = i + 1 + (n - k)
                
                if left_start >= 0 and right_end <= len(word):
                    # Construct the window
                    # word[left_start:i] + word[i+1:right_end]
                    # Note: word[i+1:right_end] has length n-k
                    # word[left_start:i] has length k
                    # Total length n
                    w = tuple(word[left_start:i] + word[i+1:right_end])
                    gained_windows.append(w)
            
            # Check if removal is valid
            # Temporarily apply changes to counter?
            # Better: check if any lost window has count 1 (and no gained window compensates?)
            # Actually, we need to see the net effect.
            
            # Net change:
            # For each lost window, count decreases by 1.
            # For each gained window, count increases by 1.
            
            # If a lost window has count 1, and it is not in gained windows (or count in gained < count lost?), 
            # it might become 0.
            
            # Simpler:
            # 1. Decrement lost, increment gained in a temp dict or just track.
            # 2. Check if any count becomes 0.
            
            # Since n is small, we can just do it.
            valid = True
            # We need to know the final count of every affected permutation.
            # Affected perms = set(lost) U set(gained)
            
            # Optimization: 
            # If a lost perm has count 1, it becomes 0 UNLESS it is also gained and the net change is >= 0?
            # No, count is global.
            # new_count[p] = old_count[p] - count_in_lost[p] + count_in_gained[p]
            # We need new_count[p] > 0 for all p.
            
            # Let's just simulate the counter update for the affected keys.
            # To avoid modifying the real counter until sure, we can check:
            # For any p in affected:
            #   if old_count[p] - lost_count[p] + gained_count[p] <= 0:
            #       invalid
            
            lost_counter = Counter(lost_windows)
            gained_counter = Counter(gained_windows)
            
            affected = set(lost_counter.keys()) | set(gained_counter.keys())
            
            for p in affected:
                old_c = perm_counts[p]
                new_c = old_c - lost_counter.get(p, 0) + gained_counter.get(p, 0)
                if new_c <= 0:
                    valid = False
                    break
            
            if valid:
                # Apply removal
                # Update counter
                for p in lost_windows:
                    perm_counts[p] -= 1
                    if perm_counts[p] == 0:
                        missing_count += 1
                for p in gained_windows:
                    if perm_counts[p] == 0:
                        missing_count -= 1
                    perm_counts[p] += 1
                
                # Remove from word
                word.pop(i)
                improved = True
                # Restart scan? 
                # Since indices shifted, continuing might miss things or use wrong indices.
                # But we are in a `for i` loop. If we pop, `word` changes.
                # It is safer to break and restart the `while` loop.
                break
                
    return "".join(map(str, word))
