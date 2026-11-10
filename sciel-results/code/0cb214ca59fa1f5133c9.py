import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    def can_add(x, current_set):
        for y in current_set:
            if x != y and conflict[x][y]:
                return False
        return True
    
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    evens = [i for i in range(1, N + 1) if i % 2 == 0]
    
    best_set = None
    best_size = 0
    
    # Greedy: odds + evens in various orders
    for trial in range(100):
        if time.time() > deadline - 60.0:
            break
        if trial == 0:
            order = sorted(evens)
        elif trial == 1:
            order = sorted(evens, reverse=True)
        else:
            order = evens[:]
            random.shuffle(order)
        
        current = odds[:]
        current_set = set(odds)
        for x in order:
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local search: try to improve
    def local_search(initial_set, max_iters=1000):
        current = list(initial_set)
        current_set = set(current)
        cur_size = len(current)
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try adding one element
            for x in range(1, N + 1):
                if x not in current_set and can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    cur_size += 1
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                continue
            
            # Try removing one element and adding others (limited)
            found = False
            for a in current:
                temp_set = current_set - {a}
                added_count = 0
                for x in range(1, N + 1):
                    if x not in temp_set and can_add(x, temp_set):
                        temp_set.add(x)
                        added_count += 1
                if added_count >= 2:
                    current = list(temp_set)
                    current_set = temp_set
                    cur_size = len(current)
                    improved = True
                    no_improve = 0
                    found = True
                    break
            
            if not improved:
                no_improve += 1
                if no_improve >= 3:
                    break
        
        return current
    
    if best_set:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Random restarts
    restart = 0
    while time.time() < deadline - 10.0:
        restart += 1
        r = random.random()
        if r < 0.4:
            # Odds + shuffled evens
            current = odds[:]
            current_set = set(odds)
            shuffled = evens[:]
            random.shuffle(shuffled)
            for x in shuffled:
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
        elif r < 0.7:
            # Full random greedy
            shuffled = list(range(1, N + 1))
            random.shuffle(shuffled)
            current = []
            current_set = set()
            for x in shuffled:
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
        else:
            # Perturb best
            current = list(best_set)
            current_set = set(current)
            for _ in range(random.randint(2, 10)):
                if current:
                    idx = random.randint(0, len(current) - 1)
                    current_set.discard(current.pop(idx))
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            random.shuffle(candidates)
            for x in candidates[:30]:
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
        
        improved = local_search(current, max_iters=500)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    return best_set
