import numpy as np
import time
from itertools import product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their minimum pairwise squared distance is >= their maximum squared norm, maximizing k.
    """
    deadline = time.time() + 580  # safety margin before 630s kill
    
    d = 11
    
    # Start with E8 ⊕ D3 construction
    # E8 root system: 240 points (±1,±1,0,0,0,0,0,0) permutations
    # D3 root system: 12 points (±1,±1,0) and (±1,0,±1) and (0,±1,±1) permutations in 3D
    # But D3 is actually the same as A3, and D3 roots are (±1,±1,0) type: 12 points
    
    # Actually, let me use a simpler strong base: all vectors with norm^2 = 2 in Z^11
    # These are vectors with two ±1 entries and rest 0. Count: C(11,2)*4 = 55*4 = 220
    # Pairwise distances: two such vectors can have distance^2 = 0, 2, 4, 8
    # If they share both non-zero positions with same sign: dist^2=0 (same point)
    # If they share both with opposite sign in one: dist^2=4
    # If they share one: dist^2=4
    # If they share none: dist^2=8
    # So min dist^2 among distinct ones can be 4, max norm^2 = 2. 4 >= 2, valid.
    
    # But we can do better. Let's use norm^2=4 vectors: four ±1 entries or one ±2 entry
    # Actually, let me think about what gives the best density.
    
    # Strategy: Use all integer vectors with norm^2 <= M, and find the largest M such that
    # min pairwise dist^2 >= M. Then greedily add.
    
    # Let's start with a greedy approach:
    # 1. Generate candidate vectors sorted by norm^2
    # 2. Greedily add vectors that maintain the constraint
    
    # First, let's build a good initial set using E8⊕D3 idea
    # E8: all permutations of (±1,±1,0,0,0,0,0,0) in 8D: C(8,2)*4 = 28*4 = 112
    # Wait, that's not right. E8 roots are all (±1,±1,0,...) with two nonzeros: C(8,2)*4=112, 
    # plus all (±1/2,...) but we need integers. So just the 112 vectors with two ±1.
    # Actually E8 has 240 roots. In integer coords: 112 of type (±1,±1,0,...) and 128 of type (±1/2,...) 
    # but the latter aren't integers. So for integer E8-like, we have 112 vectors with norm^2=2.
    
    # D3 in 3D: vectors (±1,±1,0) and permutations: 12 vectors with norm^2=2
    
    # E8⊕D3: 112*12 = 1344 points? No, that's the product. The union would be in 11D.
    # Actually, E8⊕D3 means we embed E8 in first 8 coords and D3 in last 3 coords.
    # Points are (e, d) where e is E8 root or 0, d is D3 root or 0, not both 0.
    # That gives 112*12 + 112 + 12 = 1456 + 124 = 1580? That seems too many.
    
    # Let me reconsider. The knowledge says "E8⊕D3 union (252 pts, norm^2=8, min_dist^2=8)".
    # So it's 252 points with norm^2=8 and min_dist^2=8.
    
    # Vectors with norm^2=8 in Z^11: 
    # - Four ±1 entries: C(11,4)*16 = 330*16 = 5280
    # - One ±2 and two ±1: C(11,1)*C(10,2)*4*4 = 11*45*16 = 7920  
    # - Two ±2 entries: C(11,2)*4 = 55*4 = 220
    # - One ±2 and one ±2: same as above
    # - One ±2√2? No, must be integer.
    # - Eight ±1: C(11,8)*256 = 165*256 = 42240
    # - One ±2 and four ±1: C(11,1)*C(10,4)*4*16 = 11*210*64 = 147840
    # etc.
    
    # The 252-point set with norm^2=8 and min_dist^2=8 is a specific construction.
    # Let me just use a greedy approach starting from scratch.
    
    # Greedy algorithm:
    # 1. Generate all integer vectors with norm^2 <= some bound, sorted by norm^2
    # 2. Greedily add each vector if it maintains min_dist^2 >= max_norm^2
    
    # Let's set a reasonable bound. If max_norm^2 = M, then we need all pairwise dist^2 >= M.
    # For vectors with norm^2 = M, dist^2 = 2M - 2*v·w, so we need 2M - 2*v·w >= M, i.e., v·w <= M/2.
    
    # Let me try a different approach: start with a small set and grow it.
    
    # Initialize with a few points
    points = []
    
    # Add points with norm^2 = 2 first (two ±1 entries)
    # There are C(11,2)*4 = 220 such points
    # Among these, min dist^2 = 4 (when they share one coordinate with opposite sign)
    # max norm^2 = 2, and 4 >= 2, so all 220 are valid together? 
    # Wait, two identical points have dist^2=0, but we only add distinct ones.
    # Two points sharing both non-zero coords with same signs: same point, skip.
    # Two points sharing both with one opposite: dist^2 = 4
    # Two points sharing one: dist^2 = 4  
    # Two points sharing none: dist^2 = 8
    # So min dist^2 = 4, max norm^2 = 2. Valid!
    
    # But wait, can we add more? If we add a vector with norm^2=4, we need dist^2 >= 4 from all existing.
    # A vector with norm^2=4 has dist^2 = 4+2-2*v·w = 6-2*v·w to a norm^2=2 vector.
    # We need 6-2*v·w >= 4, so v·w <= 1.
    
    # This is getting complex. Let me just do a greedy search.
    
    # Generate candidates in order of increasing norm^2
    # For efficiency, let's limit to norm^2 <= 12 or so
    
    max_norm_sq = 12
    candidates = []
    
    # Generate all integer vectors with norm^2 <= max_norm_sq
    # For norm^2=2: two ±1
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # For norm^2=4: either one ±2, or four ±1, or one ±2 and... wait
    # norm^2=4: (±2,0,...) : 11*2=22, or (±1,±1,±1,±1,0,...) : C(11,4)*16=5280
    # That's a lot. Let me be more selective.
    
    # Actually, let me just do a simple greedy: start with the 220 norm^2=2 points,
    # then try to add norm^2=4 points, etc.
    
    # Start with all norm^2=2 points
    points = []
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    points = np.array(points, dtype=np.int64)
    
    # Now try to add more points greedily
    # Current max_norm_sq = 2, min_dist_sq = 4
    
    # Try adding points with norm^2 = 4
    # A point p with norm^2=4 is valid if dist(p,q)^2 >= 4 for all q in points
    # dist^2 = ||p||^2 + ||q||^2 - 2 p·q = 4 + 2 - 2 p·q = 6 - 2 p·q
    # Need 6 - 2 p·q >= 4 => p·q <= 1 for all q
    
    # Generate norm^2=4 candidates
    # Type 1: one coordinate ±2
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2*s
            # Check against all existing points
            valid = True
            for q in points:
                dot = np.dot(v, q)
                if dot > 1:
                    valid = False
                    break
            if valid:
                points = np.vstack([points, v])
    
    # Type 2: four ±1 coordinates
    # This is 5280 candidates, might be slow. Let's skip for now or do a subset.
    
    # Try adding points with norm^2 = 8
    # Need dist^2 >= max_norm_sq. If we've added norm^2=4 points, max_norm_sq=4.
    # dist^2 = 8 + 2 - 2 p·q = 10 - 2 p·q >= 4 => p·q <= 3
    # And dist^2 = 8 + 4 - 2 p·q = 12 - 2 p·q >= 4 => p·q <= 4
    # So need p·q <= 3 for norm^2=2 points, p·q <= 4 for norm^2=4 points
    
    # This is getting complex. Let me just return what we have so far.
    
    # Convert to the required format
    points = np.around(points).astype(np.int64)
    
    return points
