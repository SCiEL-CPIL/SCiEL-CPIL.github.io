import random

def construct():
    n = 512
    
    # Precompute conflicts for each element
    # conflicts[x] contains tuples of 3 other elements that form a 4-AP with x
    conflicts = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # If we add 'a', the other three are b, c, e
            conflicts[a].append((b, c, e))
            # If we add 'b', the other three are a, c, e
            conflicts[b].append((a, c, e))
            # If we add 'c', the other three are a, b, e
            conflicts[c].append((a, b, e))
            # If we add 'e', the other three are a, b, c
            conflicts[e].append((a, b, c))

    best_set = set()
    best_size = 0
    
    # Run multiple randomized greedy trials
    # 200 trials should be safe within 60s
    for trial in range(200):
        random.seed(trial)
        in_set = [False] * (n + 1)
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        current_size = 0
        for x in elements:
            can_add = True
            for (y, z, w) in conflicts[x]:
                if in_set[y] and in_set[z] and in_set[w]:
                    can_add = False
                    break
            
            if can_add:
                in_set[x] = True
                current_size += 1
                
        if current_size > best_size:
            best_size = current_size
            best_set = set(i for i in range(1, n + 1) if in_set[i])
            
    return sorted(best_set)
