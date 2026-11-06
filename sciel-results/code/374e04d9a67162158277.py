import random
import time
import math
import numpy as np

N = 64

# Precompute all grid points as a list of (x, y) tuples
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # We'll use a greedy randomized approach with multiple restarts.
    # The key insight: a set has no isosceles triangles iff for every triple of points,
    # no two sides are equal. Equivalently, for every point P in the set and every other
    # point Q in the set, the distance PQ must be unique among all pairs involving P? 
    # No, that's not quite right. An isosceles triangle has three points A, B, C such that
    # AB = AC, or AB = BC, or AC = BC.
    
    # So when adding a new point P to an existing set S, we need to check:
    # 1. For each pair (A, B) in S, adding P must not make PA = PB (isosceles with apex P)
    # 2. For each A in S, adding P must not make PA = AA' for some... wait.
    # Actually: for each A in S, and for each B in S (B != A), we need PA != PB (apex at P)
    # Also for each pair (A, B) in S, we need AP != AB and BP != AB (apex at A or B)
    
    # Let's think more carefully. Given existing set S and new point P:
    # - For any two points A, B in S: triangle PAB is isosceles if PA=PB, or PA=AB, or PB=AB.
    #   So we need: for all A,B in S with A!=B: PA != PB and PA != AB and PB != AB.
    # - Also, we need to ensure no isosceles triangle is formed with P and one point from S?
    #   No, an isosceles triangle needs 3 points. If we have P and only one point A from S,
    #   that's not a triangle. So we only need to check triples where at least 2 points are in S
    #   and 1 is P, OR all 3 are in S (but those were already checked when they were added).
    
    # So the check for adding P to S is:
    # For all pairs (A, B) in S with A != B:
    #   Let d_PA = dist(P, A), d_PB = dist(P, B), d_AB = dist(A, B)
    #   We need: d_PA != d_PB, d_PA != d_AB, d_PB != d_AB
    
    # To make this efficient, we can precompute for each point in S the set of distances to other points in S.
    # When adding P, we need:
    # 1. All distances from P to points in S are distinct (so d_PA != d_PB for all A != B)
    # 2. For each A in S, d_PA is not equal to any distance d_AB where B is in S, B != A.
    #    Equivalently: for each A in S, d_PA is not in {d_AB : B in S, B != A}
    
    # Let's implement this with a greedy approach.
    
    def is_valid_addition(P, S):
        """Check if point P can be added to set S without forming an isosceles triangle."""
        if not S:
            return True
        if len(S) == 1:
            return True
        
        # Compute distances from P to all points in S
        dists_from_P = {}
        for A in S:
            dx = P[0] - A[0]
            dy = P[1] - A[1]
            d2 = dx*dx + dy*dy
            dists_from_P[A] = d2
        
        # Check 1: all distances from P to points in S must be distinct
        seen_dists = set()
        for A in S:
            d2 = dists_from_P[A]
            if d2 in seen_dists:
                return False
            seen_dists.add(d2)
        
        # Check 2: for each A in S, dist(P, A) must not equal dist(A, B) for any B in S, B != A
        # We need to know, for each A in S, the set of distances from A to other points in S.
        # This is expensive to compute naively. Let's precompute it.
        
        # Actually, let's precompute for each A in S the set of squared distances to other points in S.
        # But S changes as we add points. Let's maintain this incrementally.
        
        # For now, let's just compute it on the fly for correctness, and optimize later if needed.
        for A in S:
            d_PA = dists_from_P[A]
            for B in S:
                if A == B:
                    continue
                dx = A[0] - B[0]
                dy = A[1] - B[1]
                d_AB = dx*dx + dy*dy
                if d_PA == d_AB:
                    return False
        
        return True
    
    # Greedy construction with random restarts
    def greedy_construction(order):
        """Build a set by greedily adding points in the given order."""
        S = []
        for P in order:
            if is_valid_addition(P, S):
                S.append(P)
        return S
    
    # Try multiple random orderings
    num_trials = 0
    while time.time() < deadline:
        # Create a random ordering of all points
        random_order = ALL_POINTS[:]
        random.shuffle(random_order)
        
        # Also try some structured orderings
        S = greedy_construction(random_order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        num_trials += 1
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return best_set
