import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try to find a Cayley coloring for n=50
    n = 50
    # Items to assign: 1..24 (pairs) and 25 (self)
    # We represent the set of differences for each color
    # color_sets[c] is a set of integers in 1..49
    # Initially empty
    
    # To speed up, we can fix the order of assignment
    items = list(range(1, 25)) + [25]
    
    # Backtracking
    color_sets = [set() for _ in range(4)]
    
    def is_sum_free(S):
        # S is a set of ints
        # Check if a+b in S for any a,b in S
        # Optimization: if a < b, check a+b
        # Since S is symmetric, we can just check all pairs
        # For small S (max 25 elements), this is fast
        for a in S:
            for b in S:
                if (a + b) % n in S:
                    # Need to ensure a,b,c are distinct vertices?
                    # Triangle vertices x, x+a, x+a+b.
                    # If a=0, not in S.
                    # If a=b, x, x+a, x+2a. Distinct if a!=0, 2a!=0.
                    # In Z50, 2a=0 iff a=25.
                    # If a=25, b=25, sum=50=0. 0 not in S.
                    # So simple check is fine.
                    return False
        return True

    # Better check: incremental
    def check_addition(S, new_elements):
        # S is current set, new_elements is list of new ints to add
        # Check if (S U new) is sum-free
        # We only need to check sums involving at least one new element
        # Because S was sum-free
        S_new = S.union(new_elements)
        # Check pairs (a,b) where at least one is in new_elements
        # a + b in S_new
        for a in new_elements:
            for b in S_new:
                if (a + b) % n in S_new:
                    # Check if this is a valid triangle
                    # a,b,c in S_new, a+b=c
                    # Vertices 0, a, a+b
                    # If a=0, no.
                    # If a+b=0, c=0, no.
                    # So a!=0, b!=0, c!=0.
                    # This indicates a triangle.
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == len(items):
            return color_sets[:]
        
        item = items[idx]
        # The pair is {item, n-item}
        # If item == n-item (i.e., 25), it's a singleton
        pair = [item, n - item] if item != n - item else [item]
        
        for c in range(4):
            if check_addition(color_sets[c], pair):
                # Add
                for p in pair:
                    color_sets[c].add(p)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Remove
                for p in pair:
                    color_sets[c].remove(p)
        
        return None

    result = backtrack(0)
    
    if result:
        # Construct the color list
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                # Find which color set contains diff
                # diff is in 1..49
                found = False
                for c in range(4):
                    if diff in result[c]:
                        colors.append(c)
                        found = True
                        break
                if not found:
                    # Should not happen
                    colors.append(0) 
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=32
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
