# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 600
    p = 7
    q = p * p

    # Field arithmetic setup
    # alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # We map field elements to integers 0..48
    
    # Precompute multiplication and addition tables? 
    # Or just use formulas.
    # For speed in precomputation, we can use tables.
    
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute slopes
    # Directions: (1, m) for m in F_49, and (0, 1)
    # m is an integer 0..48 representing field element.
    # Vertical is index 49.
    
    # We need to map field element m to index.
    # Let's just use the integer representation 0..48 directly as index.
    # So m=0 is index 0, ..., m=48 is index 48.
    # Vertical is index 49.
    
    # Precompute dir_matrix[x, z]
    # x, z are indices 0..48 for field elements?
    # No, points are in F_49^2.
    # Point index i corresponds to (x_i, y_i) in F_49.
    # Let's represent points as (x_int, y_int) where x_int, y_int in 0..48.
    # Point index = x_int * q + y_int.
    
    # Precompute x_coord[i], y_coord[i]
    x_coords = np.zeros(q*q, dtype=np.int32)
    y_coords = np.zeros(q*q, dtype=np.int32)
    for i in range(q*q):
        x_coords[i] = i // q
        y_coords[i] = i % q

    # Precompute direction from x to z
    # dir_matrix[x_idx, z_idx] = direction index
    # If x == z, undefined (0)
    # dx = x_z - x_x
    # dy = y_z - y_y
    # If dx == 0: vertical (49)
    # Else: m = dy / dx
    
    # We need field division.
    # Let's precompute inverse of all non-zero field elements.
    inv_table = np.zeros(q, dtype=np.int32)
    for i in range(1, q):
        # Find j such that i*j = 1 mod field
        # i, j are ints 0..48
        # We can just loop or use extended euclid? 
        # Simpler: precompute multiplication table?
        # Or use the formula for inverse in F_7[alpha].
        # i = a + b*alpha.
        # Inv = (a - b*alpha) / (a^2 - 3b^2)
        
        a = i % p
        b = (i // p) % p
        denom = (a*a - 3*b*b) % p
        if denom == 0:
            # Should not happen for non-zero i in field
            inv_table[i] = 0
        else:
            denom_inv = pow(denom, p-2, p)
            real = (a * denom_inv) % p
            imag = (-b * denom_inv) % p
            inv_table[i] = to_int(real, imag)
            
    # Precompute multiplication table for speed?
    # mul_table[x, y]
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            real = (xa * ya + xb * yb * 3) % p
            imag = (xa * yb + xb * ya) % p
            mul_table[x, y] = to_int(real, imag)

    # Precompute addition/subtraction?
    # add_table[x, y]
    add_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            real = (xa + ya) % p
            imag = (xb + yb) % p
            add_table[x, y] = to_int(real, imag)
            
    sub_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            sub_table[x, y] = add_table[x, to_pair(0)[0] + to_pair(0)[1]*p if False else (p - (y%p))%p + ((p - (y//p)%p)%p)*p] # This is messy
            # Better: sub[x, y] = add[x, neg[y]]
    neg_table = np.zeros(q, dtype=np.int32)
    for y in range(q):
        ya, yb = to_pair(y)
        neg_table[y] = to_int((-ya)%p, (-yb)%p)
    for x in range(q):
        for y in range(q):
            sub_table[x, y] = add_table[x, neg_table[y]]

    # Build dir_matrix
    # Shape (q*q, q*q) might be 2401x2401 = 5.7M ints. 4 bytes each = 23MB. OK.
    dir_matrix = np.zeros((q*q, q*q), dtype=np.int32)
    
    for x_idx in range(q*q):
        xx = x_coords[x_idx]
        xy = y_coords[x_idx]
        for z_idx in range(q*q):
            if x_idx == z_idx:
                continue
            zx = x_coords[z_idx]
            zy = y_coords[z_idx]
            
            dx = sub_table[zx, xx] # zx - xx
            dy = sub_table[zy, xy] # zy - xy
            
            if dx == 0:
                dir_matrix[x_idx, z_idx] = 49
            else:
                m = mul_table[dy, inv_table[dx]]
                dir_matrix[x_idx, z_idx] = m

    # Greedy Addition Algorithm
    # State: CleanLine (Q, 50) bool, Count (Q,) int
    # C: list of points added
    
    best_C_size = 0
    best_C_points = []
    
    num_trials = 100 # Try 100 random orders
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Random order
        order = list(range(q*q))
        random.shuffle(order)
        
        CleanLine = np.ones((q*q, 50), dtype=bool)
        Count = np.full(q*q, 50, dtype=np.int32)
        C_points = []
        
        for z_idx in order:
            # Get directions from all x to z
            v_idx = dir_matrix[:, z_idx]
            
            # Check validity
            # For each x, if CleanLine[x, v_idx[x]] is True and Count[x] == 1, then invalid
            # Note: for x == z, v_idx is 0 (from init), but we should ignore x==z
            # Actually, if x==z, the line is not defined in the same way, but 
            # the condition "other points" implies x != z.
            # So we should mask out x == z.
            
            mask_x = np.ones(q*q, dtype=bool)
            mask_x[z_idx] = False
            
            # Get the specific clean status for the direction from x to z
            # CleanLine[x, v_idx[x]]
            # We can do this with advanced indexing
            clean_status = CleanLine[np.arange(q*q), v_idx]
            
            # Only consider x != z
            clean_status = clean_status & mask_x
            
            critical = (Count == 1) & clean_status
            
            if critical.any():
                continue
            
            # Valid, add z
            C_points.append(z_idx)
            
            # Update state
            # Set CleanLine[x, v_idx[x]] to False for x != z
            CleanLine[np.arange(q*q), v_idx] = False
            # But we only want to update x != z
            # The assignment above sets it for all x.
            # For x == z, v_idx is 0. We don't care about CleanLine[z, 0] 
            # because z is in C, and we don't use CleanLine for points in C 
            # for future validity checks? 
            # Wait. If z is in C, does it affect other points?
            # Yes, when checking if a NEW point w is valid, we check if 
            # any line from x to w is "dirty".
            # A line is dirty if it contains ANY point in C.
            # So if z is in C, all lines passing through z are dirty.
            # My update `CleanLine[x, v_idx] = False` correctly marks 
            # the line from x to z as dirty.
            # For x == z, the "line from z to z" is not a real line.
            # So marking CleanLine[z, 0] = False is harmless.
            
            # Update Count
            # Count[x] decreases if the line was clean.
            # clean_status was computed BEFORE update.
            # So we can just subtract clean_status.
            Count -= clean_status.astype(np.int32)
            
        if len(C_points) > best_C_size:
            best_C_size = len(C_points)
            best_C_points = C_points.copy()
            # If we reached a good size, we might break early?
            # But random orders vary. Let's do all trials if time permits.

    # Construct N
    # N = All \ C
    C_set = set(best_C_points)
    N_points = []
    for i in range(q*q):
        if i not in C_set:
            x_int = x_coords[i]
            y_int = y_coords[i]
            xa, xb = to_pair(x_int)
            ya, yb = to_pair(y_int)
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
