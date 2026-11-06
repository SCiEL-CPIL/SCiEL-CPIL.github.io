import random
import time
import math
import sys

N = 9
SIZE = 1 << N

# Precompute neighbors as lists of ints for fast iteration
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(42)
    
    best_path = [0]
    best_len = 0
    
    # Fast validity check for a full path
    def is_valid(path):
        if not path:
            return False
        n = len(path)
        if n == 1:
            return True
        # Check consecutive
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        # Check non-consecutive (chords)
        # For each vertex, it must not be adjacent to any vertex at distance >= 2
        # We can optimize by checking against the set of all vertices, but O(N^2) is fine for N=100-200
        # Actually, let's use the neighbor list property.
        # A vertex v at index i is invalid if any of its neighbors is in path[0:i-1] U path[i+2:]
        # Let's just do the bit_count check for safety, it's fast enough in Python for <200 elements
        for i in range(n):
            v = path[i]
            for j in range(n):
                if abs(i - j) > 1:
                    if (v ^ path[j]).bit_count() == 1:
                        return False
        return True

    def get_score(path):
        if not path:
            return 0
        # Assume path is a valid snake, score is number of edges
        # But we must ensure it's a simple path (no repeats)
        if len(set(path)) != len(path):
            return -1
        return len(path) - 1

    # 1. Generate diverse initial snakes using randomized greedy
    def greedy_snake(priority, start=0):
        path = [start]
        used = {start}
        cur = start
        while True:
            nxt = None
            # Try neighbors in priority order
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check if v is adjacent to any previous vertex (except cur)
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            cur = nxt
        return path

    # Generate a pool of initial snakes
    pool = []
    for i in range(50):
        if time.time() > deadline - 5:
            break
        priority = list(range(N))
        rnd.shuffle(priority)
        start = rnd.randint(0, SIZE - 1)
        p = greedy_snake(priority, start)
        if p:
            pool.append(p)
            
    # Update best
    for p in pool:
        if is_valid(p):
            s = get_score(p)
            if s > best_len:
                best_len = s
                best_path = p

    # 2. VNS: Local Search with Cut-and-Paste
    # State: A path P.
    # Moves:
    # a) Reverse a segment (if it maintains validity - tricky)
    # b) Cut a segment from P1 and paste into P2 (Splicing)
    # c) Single bit flip (if it creates a valid extension? No, we are on the path)
    
    # Since maintaining validity during local moves is hard, we will use a "Reconstruct" approach:
    # Take a prefix of a good snake, and try to extend it greedily with different priorities.
    # Also, try to merge two snakes.
    
    def extend_greedy(prefix, priority):
        path = list(prefix)
        used = set(path)
        if not path:
            return path
        cur = path[-1]
        # We need to check against all previous vertices for the new step
        # To speed up, we can keep a set of "forbidden" vertices: neighbors of all vertices in path
        # Forbidden for next step: used U neighbors_of_all_used
        # Actually, the next vertex v must not be in `used` and must not be a neighbor of any u in path[:-1]
        # So v must not be in `used` U `neighbors(path[:-1])`
        
        # Let's just do the standard check but optimized
        for _ in range(300): # Max extension steps
            if time.time() > deadline - 2:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                # Check against path[:-1]
                # Optimization: If we maintain a set of all vertices at distance 1 from path[:-1], we can check O(1)
                # But path changes. Let's just iterate.
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            cur = nxt
        return path

    # Splicing: Take prefix of P1 and suffix of P2? 
    # If P1 = [a...b] and P2 = [c...d], can we join b-c?
    # b and c must be adjacent, and the new path must be valid.
    # This is complex to validate fully.
    
    # Better VNS move: "Segment Reversal" is dangerous. 
    # Let's use "Random Restart with Memory":
    # Keep the best prefixes found.
    
    best_prefixes = []
    for p in pool:
        if is_valid(p):
            # Store prefixes of length 10, 20, ...
            for l in range(10, len(p), 10):
                best_prefixes.append(p[:l])
                
    # Local Search Loop
    while time.time() < deadline - 5:
        # Pick a base
        if best_prefixes and rnd.random() < 0.5:
            base = rnd.choice(best_prefixes)
        else:
            base = rnd.choice(pool) if pool else [0]
        
        # Try to extend with random priority
        priority = list(range(N))
        rnd.shuffle(priority)
        new_p = extend_greedy(base, priority)
        
        if is_valid(new_p):
            s = get_score(new_p)
            if s > best_len:
                best_len = s
                best_path = new_p
                # Add prefixes to memory
                for l in range(10, len(new_p), 10):
                    best_prefixes.append(new_p[:l])
                    if len(best_prefixes) > 200:
                        best_prefixes = best_prefixes[-200:]
            
            # Try "Splice": 
            # Take new_p, pick a random cut point i.
            # Prefix = new_p[:i]
            # Try to extend Prefix with a DIFFERENT random priority to see if we diverge and get longer.
            if len(new_p) > 10:
                i = rnd.randint(5, len(new_p) - 5)
                prefix = new_p[:i]
                priority2 = list(range(N))
                rnd.shuffle(priority2)
                diverged = extend_greedy(prefix, priority2)
                if is_valid(diverged):
                    s2 = get_score(diverged)
                    if s2 > best_len:
                        best_len = s2
                        best_path = diverged
                        for l in range(10, len(diverged), 10):
                            best_prefixes.append(diverged[:l])
                            if len(best_prefixes) > 200:
                                best_prefixes = best_prefixes[-200:]

    # Final check
    if is_valid(best_path):
        return best_path
    else:
        # Fallback: return a known valid short snake
        return [0]

if __name__ == "__main__":
    # Self-test
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
