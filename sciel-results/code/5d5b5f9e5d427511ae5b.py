import time
import random

def construct():
    N = 40
    M = 401
    # Safe deadline to avoid timeout
    deadline = time.time() + 110
    
    def get_edges(masks):
        edges = []
        for i in range(N):
            m = masks[i]
            for j in range(i + 1, N):
                if (m >> j) & 1:
                    edges.append([i, j])
        return edges

    def compute_books(masks):
        books = {}
        max_b = 0
        for i in range(N):
            m_i = masks[i]
            for j in range(i + 1, N):
                if (m_i >> j) & 1:
                    b = (m_i & masks[j]).bit_count()
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
        return books, max_b

    all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_masks = None
    best_max_book = float('inf')
    
    # Multiple restarts of local search
    # Each restart uses a fresh random graph
    while time.time() < deadline:
        # Initialize random graph
        masks = [0] * N
        random.shuffle(all_possible)
        for u, v in all_possible[:M]:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
        current_books, current_max_book = compute_books(masks)
        
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_masks = masks[:]
            
        # Local Search: Target the max edge
        # We try to reduce the max book size by swapping the edge with the largest book
        # with a random non-edge.
        
        # Limit the number of steps per restart to ensure progress
        max_steps = 500 
        steps = 0
        stuck_count = 0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Find the edge with the maximum book size
            max_edge = None
            max_val = -1
            for e, b in current_books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_val < best_max_book:
                best_max_book = max_val
                best_masks = masks[:]
                
            # If we hit the theoretical lower bound (approx 7), we can stop early
            if max_val <= 7:
                break
                
            u, v = max_edge
            improved = False
            
            # Try several random swaps
            for _ in range(20):
                x = random.randint(0, N - 1)
                y = random.randint(0, N - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if x == u and y == v:
                    continue
                # Check if (x, y) is a non-edge
                if (masks[x] >> y) & 1:
                    continue
                
                # Perform Swap: Remove (u, v), Add (x, y)
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                
                new_books, new_max_book = compute_books(masks)
                
                if new_max_book < max_val:
                    # Improvement
                    improved = True
                    current_books = new_books
                    max_val = new_max_book
                    stuck_count = 0
                    break
                elif new_max_book == max_val:
                    # Neutral move, accept with some probability to explore
                    if random.random() < 0.5:
                        improved = True
                        current_books = new_books
                        # max_val stays same
                        stuck_count += 1
                        break
                    else:
                        # Revert
                        masks[x] ^= (1 << y)
                        masks[y] ^= (1 << x)
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                else:
                    # Worse, revert
                    masks[x] ^= (1 << y)
                    masks[y] ^= (1 << x)
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
            
            if not improved:
                stuck_count += 1
                if stuck_count > 20:
                    break # Stuck, restart
            
            # If we made a neutral move, we might want to increase stuck_count slightly
            # But the logic above handles it.
            
    return get_edges(best_masks)
