import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s safety margin

    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
        
        perms, seen = [], set()
        # Extract all distinct permutations of length m-1 from the current word
        # that consist of distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            # Optimization: The overlap is usually large
            # We can check from max possible down
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check time
        if time.time() > deadline:
            break

    return "".join(map(str, word))
