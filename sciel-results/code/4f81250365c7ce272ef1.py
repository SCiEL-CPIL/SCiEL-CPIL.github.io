import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 30
    
    # Precompute all 6-tuples? No, too many.
    # But we can precompute the list of 4 other vertices for each pair (u,v)?
    # Or just generate them on the fly.
    
    # Let's use a list of all 6-tuples to make it easier to iterate?
    # 593,775 tuples. Storing them as a list of tuples takes memory but is okay.
    # Actually, we don't need to store all. We just need to check the ones affected.
    
    # We maintain a count of bad tuples?
    # Or just check if any bad tuple exists.
    
    # Strategy:
    # 1. Initialize adj.
    # 2. Compute initial bad count and list of bad tuples?
    #    Computing initial list takes 600k * 15 = 9M ops. ~1-2s.
    # 3. Loop:
    #    a. If bad list is empty, return.
    #    b. Pick a random bad tuple from the list.
    #    c. Flip an edge in it.
    #    d. Update the bad list:
    #       - Remove tuples that are no longer bad?
    #       - Add tuples that became bad?
    #       This is complex.
    
    # Simpler:
    # Just re-scan the 20,475 affected tuples after each flip.
    # If any is bad, we continue.
    # But we need to know if the *whole* graph is valid.
    # So we must ensure no *other* bad tuples exist.
    # So we need a global count or a flag.
    
    # Let's maintain a set of bad tuples?
    # 600k items in a set is large but manageable.
    # But updating it is hard.
    
    # Alternative:
    # Just do a full scan every K steps?
    # No, we want to converge.
    
    # Let's try this:
    # 1. Full scan to get a list of bad tuples.
    # 2. If empty, done.
    # 3. Pick one, flip edge.
    # 4. Full scan again? No, too slow.
    
    # How about:
    # Keep a count of bad tuples.
    # When flipping (u,v):
    #   Identify all 6-tuples containing u,v. (20,475)
    #   For each, check if it was bad before, and is bad now.
    #   Update count.
    #   If count becomes 0, done.
    
    # This requires knowing if a tuple was bad before.
    # We can store the status of each tuple?
    # 600k booleans. Easy.
    
    # Let's index tuples by their sorted vertices?
    # Or just use a dictionary mapping tuple to status?
    # 600k entries.
    
    # Let's try to implement this.
    
    # Precompute all 6-tuples
    # This takes time to generate.
    # itertools.combinations(range(30), 6)
    
    # Let's see if we can do it.
    
    def get_all_tuples(n):
        import itertools
        return list(itertools.combinations(range(n), 6))
        
    # 593,775 tuples.
    # Generation takes a second or two.
    
    # Let's try.
    
    all_tuples = get_all_tuples(n)
    # Map tuple to index? Or just use the tuple as key in a dict?
    # Dict with 600k keys is okay.
    
    # Status: 1 if bad, 0 if good.
    status = {}
    bad_count = 0
    
    def is_bad(verts, adj):
        c0 = 0
        v0, v1, v2, v3, v4, v5 = verts
        if adj[v0][v1] == 0: c0 += 1
        if adj[v0][v2] == 0: c0 += 1
        if adj[v0][v3] == 0: c0 += 1
        if adj[v0][v4] == 0: c0 += 1
        if adj[v0][v5] == 0: c0 += 1
        if adj[v1][v2] == 0: c0 += 1
        if adj[v1][v3] == 0: c0 += 1
        if adj[v1][v4] == 0: c0 += 1
        if adj[v1][v5] == 0: c0 += 1
        if adj[v2][v3] == 0: c0 += 1
        if adj[v2][v4] == 0: c0 += 1
        if adj[v2][v5] == 0: c0 += 1
        if adj[v3][v4] == 0: c0 += 1
        if adj[v3][v5] == 0: c0 += 1
        if adj[v4][v5] == 0: c0 += 1
        return c0 >= 12 or (15 - c0) >= 12

    # Initialize
    adj = np.random.randint(0, 2, (n, n))
    adj = (adj + adj.T) % 2
    np.fill_diagonal(adj, 0)
    
    for t in all_tuples:
        if is_bad(t, adj):
            status[t] = 1
            bad_count += 1
        else:
            status[t] = 0
            
    # Search
    for step in range(5000):
        if time.time() > deadline:
            break
            
        if bad_count == 0:
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(int(adj[i][j]))
            return {'n': n, 'colors': colors}
            
        # Pick a random bad tuple
        # We need a list of bad tuples to pick from
        # Maintaining a list of bad tuples is efficient.
        # Let's maintain a list `bad_tuples`.
        
        # Re-initialize with list
        # This is getting complex. Let's just pick a random tuple from all_tuples
        # and check if it's bad?
        # No, that's inefficient if bad_count is small.
        
        # Let's maintain a list of bad tuples.
        
    # This approach is getting messy in my head.
    # Let's go back to the simple "Full Scan" but optimized with NumPy?
    
    # Actually, 600k tuples is small.
    # If we just do a full scan every 10 steps?
    # 10 steps * 20k checks = 200k checks.
    # Full scan = 600k checks.
    # So every 10 steps, we do a full scan to verify.
    # In between, we just flip edges in known bad tuples.
    
    # Let's try a simpler heuristic:
    # 1. Full scan to get list of bad tuples.
    # 2. If empty, done.
    # 3. Pick random bad tuple.
    # 4. Flip random edge in it.
    # 5. Repeat 3-4 for 100 steps.
    # 6. Full scan again.
    
    # This is robust.
    
    for restart in range(5):
        if time.time() > deadline: break
        
        adj = np.random.randint(0, 2, (n, n))
        adj = (adj + adj.T) % 2
        np.fill_diagonal(adj, 0)
        
        for phase in range(20):
            if time.time() > deadline: break
            
            # Find all bad tuples
            bad_tuples = []
            for t in all_tuples:
                if is_bad(t, adj):
                    bad_tuples.append(t)
                    
            if not bad_tuples:
                colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        colors.append(int(adj[i][j]))
                return {'n': n, 'colors': colors}
            
            # Flip edges in some bad tuples
            for _ in range(100):
                if time.time() > deadline: break
                t = random.choice(bad_tuples)
                i, j = random.sample(t, 2)
                adj[i][j] = 1 - adj[i][j]
                adj[j][i] = adj[i][j]
                
    # Fallback n=29
    n = 29
    residues = set()
    for x in range(1, 29):
        residues.add((x * x) % 29)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
