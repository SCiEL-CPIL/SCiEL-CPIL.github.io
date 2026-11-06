import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n = 32
    graph = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
            graph[i][j] = c
            graph[j][i] = c
            
    def count_conflicts_2d(graph, n):
        count = 0
        for i in range(n-2):
            for j in range(i+1, n-1):
                c_ij = graph[i][j]
                for k in range(j+1, n):
                    if c_ij == graph[i][k] and c_ij == graph[j][k]:
                        count += 1
        return count

    def fix_conflicts_2d(graph, n, time_limit, max_steps=20000):
        conflicts = count_conflicts_2d(graph, n)
        steps = 0
        while conflicts > 0 and steps < max_steps:
            if time.time() > time_limit:
                return False
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                steps += 1
                continue
            
            old_c = graph[u][v]
            best_delta = 1
            best_c = None
            for c in range(4):
                if c == old_c: continue
                delta = 0
                for w in range(n):
                    if w == u or w == v: continue
                    c_uw = graph[u][w]
                    c_vw = graph[v][w]
                    was_mono = (old_c == c_uw and old_c == c_vw)
                    is_mono = (c == c_uw and c == c_vw)
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_c = c
                    
            if best_c is not None and best_delta < 0:
                graph[u][v] = best_c
                graph[v][u] = best_c
                conflicts += best_delta
            else:
                if random.random() < 0.05:
                    new_c = random.randint(0, 3)
                    if new_c != old_c:
                        delta = 0
                        for w in range(n):
                            if w == u or w == v: continue
                            c_uw = graph[u][w]
                            c_vw = graph[v][w]
                            was_mono = (old_c == c_uw and old_c == c_vw)
                            is_mono = (new_c == c_uw and new_c == c_vw)
                            if was_mono and not is_mono:
                                delta -= 1
                            elif not was_mono and is_mono:
                                delta += 1
                        graph[u][v] = new_c
                        graph[v][u] = new_c
                        conflicts += delta
            steps += 1
        return conflicts == 0

    for target_n in range(33, 51):
        if time.time() > deadline - 5:
            break
        
        success = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Resize
            new_graph = [[-1] * target_n for _ in range(target_n)]
            for i in range(n):
                for j in range(n):
                    new_graph[i][j] = graph[i][j]
            graph = new_graph
            
            new_v = n
            for j in range(n):
                c = random.randint(0, 3)
                graph[new_v][j] = c
                graph[j][new_v] = c
            
            n = target_n
            
            if fix_conflicts_2d(graph, n, deadline - 5):
                success = True
                break
            else:
                # Revert to previous state? 
                # It's easier to just keep the failed state and try again? 
                # No, the graph is modified. We should restart from the valid n-1 state.
                # So we need to save the previous valid graph.
                pass
        
        if not success:
            # If we failed to grow, we return the previous valid n
            # But we lost the previous valid graph state because we modified 'graph'.
            # We should keep a copy of the last valid graph.
            break

    # If we broke out early, we need the last valid graph.
    # The logic above modifies 'graph' in place.
    # If fix_conflicts fails, 'graph' is corrupted.
    # We must keep a snapshot.
    
    # Let's refactor to keep `last_valid_graph`.
    
    # Re-run with snapshot logic
    last_valid_n = 32
    last_valid_graph = [row[:] for row in graph]
    
    for target_n in range(33, 51):
        if time.time() > deadline - 5:
            break
        
        success = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Restore from last valid
            graph = [row[:] for row in last_valid_graph]
            
            # Resize
            new_graph = [[-1] * target_n for _ in range(target_n)]
            for i in range(last_valid_n):
                for j in range(last_valid_n):
                    new_graph[i][j] = graph[i][j]
            graph = new_graph
            
            new_v = last_valid_n
            for j in range(last_valid_n):
                c = random.randint(0, 3)
                graph[new_v][j] = c
                graph[j][new_v] = c
            
            if fix_conflicts_2d(graph, target_n, deadline - 5):
                last_valid_n = target_n
                last_valid_graph = [row[:] for row in graph]
                success = True
                break
                
        if not success:
            break
            
    n = last_valid_n
    colors_list = []
    for i in range(n):
        for j in range(i+1, n):
            colors_list.append(last_valid_graph[i][j])
            
    return {'n': n, 'colors': colors_list}
