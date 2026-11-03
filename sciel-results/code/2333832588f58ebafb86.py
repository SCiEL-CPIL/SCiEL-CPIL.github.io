import time
from collections import Counter, defaultdict
import heapq

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # 1. Generate standard superpermutation via recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if time.time() - start_time > deadline:
        return "".join(map(str, word))
    
    # 2. Optimized greedy deletion
    seq = word[:]
    L = len(seq)
    
    # Count occurrences of each n-window
    window_counts = Counter()
    for i in range(L - n + 1):
        w = tuple(seq[i:i+n])
        window_counts[w] += 1
    
    # For each position, track how many windows it belongs to and their counts
    # pos_window_info[pos] = list of (window_start, window_tuple)
    # We'll use a more efficient approach: track for each position the min count of windows containing it
    
    def compute_min_count_for_position(seq, n, window_counts):
        """For each position, compute the minimum count among all windows containing it."""
        L = len(seq)
        min_counts = [float('inf')] * L
        
        for start in range(L - n + 1):
            w = tuple(seq[start:start+n])
            cnt = window_counts[w]
            for pos in range(start, start + n):
                if cnt < min_counts[pos]:
                    min_counts[pos] = cnt
        
        return min_counts
    
    def try_delete_positions(seq, n, window_counts, min_counts, max_deletions=10000):
        """Try to delete positions where min_count > 1. Returns new seq, window_counts, min_counts, deletions made."""
        deletions = 0
        L = len(seq)
        
        # Find all positions that can be deleted (min_count > 1)
        # Use a priority queue: prefer positions with lower min_count (easier to delete)
        # and positions near the edges (less impact)
        pq = []
        for i in range(L):
            if min_counts[i] > 1:
                # Priority: lower min_count first, then by position (edges first)
                edge_dist = min(i, L - 1 - i)
                heapq.heappush(pq, (min_counts[i], edge_dist, i))
        
        while pq and deletions < max_deletions:
            if time.time() - start_time > deadline:
                break
            
            mc, ed, i = heapq.heappop(pq)
            
            # Verify this position is still deletable
            if i >= len(seq) or min_counts[i] <= 1:
                continue
            
            # Check if all windows containing position i have count > 1
            start_min = max(0, i - n + 1)
            start_max = min(len(seq) - n, i)
            
            can_del = True
            affected_windows = []
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                if window_counts[w] <= 1:
                    can_del = False
                    break
                affected_windows.append((start, w))
            
            if not can_del:
                continue
            
            # Perform deletion
            for start, w in affected_windows:
                window_counts[w] -= 1
                if window_counts[w] == 0:
                    del window_counts[w]
            
            seq.pop(i)
            deletions += 1
            
            # Update min_counts for affected positions
            # Positions near i are affected
            L_new = len(seq)
            update_start = max(0, i - n)
            update_end = min(L_new, i + n)
            
            # Recompute min_counts for positions in the update range
            for pos in range(update_start, update_end):
                if pos >= L_new:
                    continue
                new_min = float('inf')
                p_start_min = max(0, pos - n + 1)
                p_start_max = min(L_new - n, pos)
                for start in range(p_start_min, p_start_max + 1):
                    w = tuple(seq[start:start+n])
                    cnt = window_counts.get(w, 0)
                    if cnt < new_min:
                        new_min = cnt
                min_counts[pos] = new_min if new_min != float('inf') else 0
                
                # Add to queue if deletable
                if min_counts[pos] > 1:
                    edge_dist = min(pos, L_new - 1 - pos)
                    heapq.heappush(pq, (min_counts[pos], edge_dist, pos))
        
        return seq, window_counts, min_counts, deletions
    
    # Multiple passes with increasing aggressiveness
    min_counts = compute_min_count_for_position(seq, n, window_counts)
    
    for pass_num in range(5):
        if time.time() - start_time > deadline:
            break
        
        max_del = 50000 if pass_num == 0 else 20000
        seq, window_counts, min_counts, dels = try_delete_positions(
            seq, n, window_counts, min_counts, max_deletions=max_del
        )
        
        if dels == 0:
            break
        
        # Recompute min_counts from scratch every few passes for accuracy
        if pass_num % 2 == 1:
            min_counts = compute_min_count_for_position(seq, n, window_counts)
    
    return "".join(map(str, seq))
