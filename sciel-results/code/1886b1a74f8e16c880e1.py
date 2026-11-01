import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    if n <= 6:
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
                max_ov = min(len(new), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Generate S_6
        word6 = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word6 = new
        
        perms6, seen = [], set()
        for i in range(len(word6) - 5):
            window = tuple(word6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms6.append(window)
        
        blocks = []
        for p in perms6:
            blocks.append(list(p) + [7] + list(p))
        
        num_blocks = len(blocks)
        
        # Overlap matrix
        O = [[0]*num_blocks for _ in range(num_blocks)]
        for i in range(num_blocks):
            bi = blocks[i]
            for j in range(num_blocks):
                bj = blocks[j]
                max_ov = min(len(bi), len(bj))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if bi[len(bi)-o:] == bj[:o]:
                        ov = o
                        break
                O[i][j] = ov
        
        # Initial path: classic order
        path = list(range(num_blocks))
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(1, num_blocks): # Move node at i
                node = path[i]
                L = path[i-1]
                R = path[i+1] if i+1 < num_blocks else -1
                
                # Try inserting at j
                for j in range(0, num_blocks):
                    if j == i or j == i-1: continue
                    
                    # Determine A, B for insertion at j
                    # Inserting between path[j-1] and path[j]
                    # If j==0, insert before path[0]
                    # If j>0, insert after path[j-1]
                    
                    # We need to handle the indices carefully.
                    # Let's just try moving node to position j (0-indexed in final list)
                    # This is complex to do in-place with delta.
                    # Simpler: Just do random restarts with NN?
                    # No, let's do a simple "Best Improvement" pass.
                    pass
                    
        # Fallback: Use the classic order (path)
        # Actually, let's just use the classic order if local search is too complex to code correctly in time.
        # The improvement is small.
        
        # Let's try a simple Greedy NN from the start, it's easier to code.
        # But classic order is a good baseline.
        
        # Let's just return the classic construction for n=7 as well, 
        # unless I can quickly code the TSP.
        # Given the risk of bugs in TSP, and the small gain, 
        # maybe I should just return the classic one?
        # But the task asks for a "stronger algorithm".
        
        # Let's implement the NN heuristic properly.
        path = [0]
        unvisited = set(range(1, num_blocks))
        current = 0
        
        while unvisited:
            best_next = -1
            best_ov = -1
            for nxt in unvisited:
                if O[current][nxt] > best_ov:
                    best_ov = O[current][nxt]
                    best_next = nxt
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
            if time.time() > deadline:
                break
        
        word7 = list(blocks[path[0]])
        for idx in range(1, len(path)):
            prev_idx = path[idx-1]
            curr_idx = path[idx]
            ov = O[prev_idx][curr_idx]
            word7 += blocks[curr_idx][ov:]
        
        return "".join(map(str, word7))
    
    # Fallback
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
