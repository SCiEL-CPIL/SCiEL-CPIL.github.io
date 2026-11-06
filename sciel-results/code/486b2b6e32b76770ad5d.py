# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 128
    # Precompute LCMs
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
            
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N):
        for b in range(a+1, N):
            L_ab = lcms[a-1][b-1]
            for c in range(b+1, N):
                if lcms[a-1][c-1] == L_ab and lcms[b-1][c-1] == L_ab:
                    bad_triples.append((a, b, c))
                    
    # Helper: Check if a set is valid (no bad triples)
    def is_valid(S):
        for t in bad_triples:
            if t[0] in S and t[1] in S and t[2] in S:
                return False
        return True
        
    # Helper: Get all bad triples within S
    def get_active_triples(S):
        return [t for t in bad_triples if t[0] in S and t[1] in S and t[2] in S]

    # Phase 1: Greedy Remove
    S = set(range(1, N+1))
    
    while True:
        active_triples = get_active_triples(S)
        if not active_triples:
            break
            
        counts = {}
        for t in active_triples:
            for x in t:
                counts[x] = counts.get(x, 0) + 1
                
        max_count = -1
        victim = None
        for x, c in counts.items():
            if c > max_count:
                max_count = c
                victim = x
                
        S.remove(victim)
        
    # Phase 2: Local Search / Recovery
    # Try to add back elements that were removed, if they don't cause conflicts
    # Or try to swap? 
    # Simple: Try to add any element from 1..128 not in S.
    # If it causes conflict, skip.
    # To be smarter: If adding x causes conflict with triples T, 
    # can we remove one element from T to fix it? 
    # This is complex. Let's stick to simple addition first.
    
    for x in range(1, N+1):
        if x in S:
            continue
        # Check if adding x makes S invalid
        # We only need to check triples containing x
        # Precompute triples_by_element
        # Let's just check all active triples of S U {x}
        # Optimization: Check triples involving x
        # For each bad triple (a,b,c) where x is one of a,b,c:
        # If the other two are in S, then adding x creates a conflict.
        
        conflict = False
        for t in bad_triples:
            if x in t:
                other_two = [y for y in t if y != x]
                if other_two[0] in S and other_two[1] in S:
                    conflict = True
                    break
        
        if not conflict:
            S.add(x)
            
    return sorted(S)
# EVOLVE-BLOCK-END
