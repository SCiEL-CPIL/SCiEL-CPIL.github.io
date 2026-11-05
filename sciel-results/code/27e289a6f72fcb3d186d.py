import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    max_edges = n * (n - 1) // 2
    colors = 20
    
    # State representation
    # adj[u][v] = color, or None if no edge
    # We use a list of dicts for sparse access, or a 2D array for fast indexing?
    # 2D array is faster for checks.
    # adj[u][v] stores color + 1 (1..20) or 0 for no edge.
    
    def is_valid_full(adj):
        """Check if the whole graph is valid. O(n^4) roughly."""
        # Count bad 4-cycles
        bad = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] == 0:
                    continue
                # Check all 4-cycles containing (u,v)
                # Path u-a-b-v
                for a in range(n):
                    if a == u or a == v: continue
                    if adj[u][a] == 0: continue
                    for b in range(n):
                        if b == u or b == v or b == a: continue
                        if adj[v][b] == 0: continue
                        if adj[a][b] == 0: continue
                        # Cycle edges: (u,v), (v,b), (b,a), (a,u)
                        c1 = adj[u][v]
                        c2 = adj[v][b]
                        c3 = adj[a][b]
                        c4 = adj[u][a]
                        if len({c1, c2, c3, c4}) < 4:
                            bad += 1
        return bad == 0

    def get_bad_count_delta(adj, u, v):
        """
        Returns the number of BAD 4-cycles involving edge (u,v).
        Assumes (u,v) is currently NOT in the graph.
        If we were to add it, how many 4-cycles would be invalid?
        Note: This checks if the 3 existing edges (u,a), (a,b), (b,v) have distinct colors.
        If they do, the cycle IS valid regardless of c(u,v) (we can pick a 4th color).
        If they don't, the cycle is INVALID for any c(u,v).
        So this function counts "structurally blocked" 4-cycles.
        """
        bad = 0
        for a in range(n):
            if a == u or a == v: continue
            if adj[u][a] == 0: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if adj[v][b] == 0: continue
                if adj[a][b] == 0: continue
                # Check colors of (u,a), (a,b), (b,v)
                c1 = adj[u][a]
                c2 = adj[a][b]
                c3 = adj[v][b]
                if len({c1, c2, c3}) < 3:
                    bad += 1
        return bad

    def can_add_edge(adj, u, v):
        """
        Check if edge (u,v) can be added to make a valid graph.
        Returns True if possible, False otherwise.
        """
        # 1. Check for structural blocks (duplicate colors in the 3-path)
        # If any path u-a-b-v has c(u,a)==c(a,b) or c(u,a)==c(b,v) or c(a,b)==c(b,v),
        # then we CANNOT add (u,v) because that 4-cycle can never be rainbow.
        for a in range(n):
            if a == u or a == v: continue
            if adj[u][a] == 0: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if adj[v][b] == 0: continue
                if adj[a][b] == 0: continue
                c1 = adj[u][a]
                c2 = adj[a][b]
                c3 = adj[v][b]
                if c1 == c2 or c1 == c3 or c2 == c3:
                    return False
        # 2. Check if there is a valid color for (u,v)
        # We need c(u,v) != c(u,a), c(u,v) != c(a,b), c(u,v) != c(b,v) for ALL such paths.
        forbidden = set()
        for a in range(n):
            if a == u or a == v: continue
            if adj[u][a] == 0: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if adj[v][b] == 0: continue
                if adj[a][b] == 0: continue
                forbidden.add(adj[u][a])
                forbidden.add(adj[a][b])
                forbidden.add(adj[v][b])
        
        if len(forbidden) >= 20:
            return False
        return True

    def find_valid_color(adj, u, v):
        """Find a color for (u,v) that makes all involved 4-cycles rainbow."""
        forbidden = set()
        for a in range(n):
            if a == u or a == v: continue
            if adj[u][a] == 0: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if adj[v][b] == 0: continue
                if adj[a][b] == 0: continue
                forbidden.add(adj[u][a])
                forbidden.add(adj[a][b])
                forbidden.add(adj[v][b])
        
        for c in range(1, 21):
            if c not in forbidden:
                return c
        return 0

    def random_valid_graph():
        """Generate a random valid graph using greedy addition."""
        adj = [[0]*n for _ in range(n)]
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline - 5:
                break
            if can_add_edge(adj, u, v):
                c = find_valid_color(adj, u, v)
                adj[u][v] = c
                adj[v][u] = c
        return adj

    # Initialize with a greedy graph
    best_adj = random_valid_graph()
    best_count = sum(1 for u in range(n) for v in range(u+1, n) if best_adj[u][v] > 0)
    
    # Local Search / Simulated Annealing
    # We will try to improve the graph by adding edges or recoloring.
    # Since we must maintain validity, we only accept moves that result in a valid graph.
    # To escape local minima, we might need to remove edges temporarily?
    # No, "Maximize edges" implies we should only add or swap if it leads to a denser graph.
    # But if we are stuck, removing an edge allows a different configuration.
    # However, removing an edge decreases the score immediately.
    # Let's try "Recoloring" first, as it doesn't change edge count but might unblock additions.
    
    current_adj = [row[:] for row in best_adj]
    current_count = best_count
    
    # Precompute all possible edges for random selection
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    temp = 1000.0
    cooldown = 0.999
    
    steps = 0
    while time.time() < deadline - 10:
        # Random move
        move_type = random.choice(['add', 'recolor', 'remove'])
        
        if move_type == 'add':
            # Pick a random missing edge
            attempts = 0
            while attempts < 50:
                u, v = random.choice(all_edges)
                if current_adj[u][v] == 0:
                    break
                attempts += 1
            
            if attempts >= 50:
                continue
                
            if can_add_edge(current_adj, u, v):
                c = find_valid_color(current_adj, u, v)
                if c > 0:
                    current_adj[u][v] = c
                    current_adj[v][u] = c
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [row[:] for row in current_adj]
                        temp = 1000.0 # Reset temp on improvement
                    else:
                        temp *= cooldown
                        
        elif move_type == 'recolor':
            # Pick a random existing edge
            attempts = 0
            while attempts < 50:
                u, v = random.choice(all_edges)
                if current_adj[u][v] > 0:
                    break
                attempts += 1
            
            if attempts >= 50:
                continue
                
            old_c = current_adj[u][v]
            # Try all colors except old_c
            new_colors = [c for c in range(1, 21) if c != old_c]
            random.shuffle(new_colors)
            
            for new_c in new_colors:
                # Check if changing color to new_c keeps graph valid
                # Only 4-cycles involving (u,v) are affected.
                # We need to ensure that for all paths u-a-b-v, the 4 colors are distinct.
                valid = True
                for a in range(n):
                    if a == u or a == v: continue
                    if current_adj[u][a] == 0: continue
                    for b in range(n):
                        if b == u or b == v or b == a: continue
                        if current_adj[v][b] == 0: continue
                        if current_adj[a][b] == 0: continue
                        c1 = new_c
                        c2 = current_adj[v][b]
                        c3 = current_adj[a][b]
                        c4 = current_adj[u][a]
                        if len({c1, c2, c3, c4}) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_adj[u][v] = new_c
                    current_adj[v][u] = new_c
                    # Recoloring doesn't change count, so no best update
                    # But it might enable future adds.
                    # Accept with some probability to allow exploration?
                    # Since it's neutral on score, let's accept if it doesn't hurt.
                    # It doesn't hurt validity.
                    # We accept it to diversify.
                    break # Stop after first valid recolor found? Or pick random valid?
                    # Let's pick a random valid recolor to avoid bias.
                    # For now, first valid is fine.

        elif move_type == 'remove':
            # Only remove if we are stuck? Or always allow to escape?
            # Removing an edge reduces count.
            # We should only do this if we haven't improved in a while?
            # Or use Simulated Annealing logic: accept with prob exp(-1/T).
            # Since score drops by 1, and we want max, we should be careful.
            # Let's only remove if best_count hasn't changed for a long time?
            # Simple heuristic: Remove with low probability.
            if random.random() < 0.1:
                attempts = 0
                while attempts < 50:
                    u, v = random.choice(all_edges)
                    if current_adj[u][v] > 0:
                        break
                    attempts += 1
                if attempts < 50:
                    current_adj[u][v] = 0
                    current_adj[v][u] = 0
                    current_count -= 1

    # Final check and extraction
    # Verify best_adj is valid (should be, but safety)
    # If best_adj is not valid (bug?), we need a fallback.
    # But our moves preserve validity (add: checked, recolor: checked, remove: preserves validity).
    # So best_adj is valid.
    
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] > 0:
                final_edges.append([u, v, best_adj[u][v]])
    
    return final_edges

