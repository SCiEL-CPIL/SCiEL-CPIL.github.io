import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Optimization: Local Deletion Search
    all_perms = set(itertools.permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in all_perms:
                counts[p] = counts.get(p, 0) + 1
        return counts

    counts = get_counts(word)
    
    def is_valid(counts):
        for p in all_perms:
            if counts.get(p, 0) == 0:
                return False
        return True
        
    if not is_valid(counts):
        return "".join(map(str, word))
        
    # Efficient deletion: scan right-to-left to avoid index shifting issues
    # After deletion at position i, positions < i are unaffected
    # We can continue scanning left from i-1
    
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        # Scan from right to left
        for i in range(len(word) - 2, 0, -1):
            if time.time() > deadline - 5:
                return "".join(map(str, word))
            
            # Windows covering position i: starts in [i-n+1, i], valid: [0, len(word)-n]
            start = max(0, i - n + 1)
            end = min(i, len(word) - n)
            
            affected = []
            valid_deletion = True
            
            for j in range(start, end + 1):
                p = tuple(word[j:j+n])
                if p in all_perms:
                    affected.append((j, p))
                    counts[p] -= 1
                    if counts[p] == 0:
                        valid_deletion = False
                        break
            
            if valid_deletion:
                word.pop(i)
                improved = True
                # Recompute counts efficiently: only windows near position i changed
                # But for correctness, recompute all (it's O(L*n) which is fine for L~6000)
                counts = get_counts(word)
                # Continue scanning left from i-1 (since we're going right-to-left,
                # we just let the for loop continue to i-1)
                # But we need to adjust: after pop at i, the next index to check is i-1
                # The for loop will naturally go to i-1 next, which is correct
            else:
                # Revert
                for j, p in affected:
                    counts[p] += 1

    return "".join(map(str, word))
