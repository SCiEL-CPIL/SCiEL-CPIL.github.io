import numpy as np
import time
import science_chemistry

N = 38

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Generate FCC lattice points
    # FCC points in cubic integer coordinates: x+y+z is even.
    # We need to generate enough points to pick the 38 closest.
    # Max distance for 38th point:
    # Shell 0: 1 (dist 0)
    # Shell 1: 12 (dist sqrt(2))
    # Shell 2: 6 (dist 2)
    # Shell 3: 24 (dist sqrt(6))
    # Total 1+12+6+24 = 43. So we need points up to dist sqrt(6).
    # Integer coords up to 3 should be sufficient.
    
    points = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-3, 4):
                if (x + y + z) % 2 == 0:
                    points.append((x, y, z))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Scale factor to make nearest neighbor (dist sqrt(2)) equal to 1.12
    scale = 1.12 / np.sqrt(2)
    
    candidates = []
    
    # Candidate 1: 38 most central FCC points
    if len(points) >= 38:
        pos1 = np.array(points[:38], dtype=float) * scale
        candidates.append(pos1)
        
    # Candidate 2: 37-atom truncated octahedron + 1 atom at (2,0,0)
    # The 37-atom cluster is 1 + 12 (dist sqrt(2)) + 24 (dist sqrt(6)).
    # It excludes the 6 atoms at dist 2.
    # We construct this explicitly to preserve symmetry.
    octa = []
    # 12 atoms at (1,1,0) permutations
    for perm in [(1,1,0), (1,0,1), (0,1,1), (-1,1,0), (-1,0,1), (0,-1,1), 
                 (1,-1,0), (1,0,-1), (0,1,-1), (-1,-1,0), (-1,0,-1), (0,-1,-1)]:
        octa.append(perm)
        
    cubic = []
    # 24 atoms at (2,1,1) permutations
    for x in [-2, 2]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                # Permutations of (2,1,1)
                # We need to generate all unique permutations of (2,1,1) and (-2,1,1) etc.
                pass
    
    # Easier: Filter the sorted points list to get the specific shells
    # Shell 0: dist^2 = 0
    # Shell 1: dist^2 = 2
    # Shell 2: dist^2 = 4
    # Shell 3: dist^2 = 6
    
    shell0 = [p for p in points if p[0]**2 + p[1]**2 + p[2]**2 == 0]
    shell1 = [p for p in points if p[0]**2 + p[1]**2 + p[2]**2 == 2]
    shell2 = [p for p in points if p[0]**2 + p[1]**2 + p[2]**2 == 4]
    shell3 = [p for p in points if p[0]**2 + p[1]**2 + p[2]**2 == 6]
    
    # 37-atom truncated octahedron: Shell0 + Shell1 + Shell3
    # Note: Shell3 has 24 atoms. 1+12+24 = 37.
    base_37 = shell0 + shell1 + shell3
    
    if len(base_37) == 37:
        # Add 38th atom from Shell2 (dist 2)
        if shell2:
            pos2 = np.array(base_37 + [shell2[0]], dtype=float) * scale
            candidates.append(pos2)
            
        # Add 38th atom from Shell3? No, Shell3 is full in base_37.
        # Add 38th atom from a "hole"? 
        # Let's try adding an atom from Shell2 at a different position
        if len(shell2) > 1:
            pos3 = np.array(base_37 + [shell2[1]], dtype=float) * scale
            candidates.append(pos3)
            
        # Add 38th atom from Shell3? 
        # Wait, if we add from Shell3, we duplicate? No, we just pick one from Shell3 to add to a smaller base?
        # No, base_37 already has all 24 from Shell3.
        # So we must add from Shell2 or further out.
        # Let's try adding from Shell4 (dist^2 = 8? (2,2,0) sum 4 even. Dist^2=8).
        shell4 = [p for p in points if p[0]**2 + p[1]**2 + p[2]**2 == 8]
        if shell4:
            pos4 = np.array(base_37 + [shell4[0]], dtype=float) * scale
            candidates.append(pos4)

    best_energy = float('inf')
    best_pos = candidates[0] if candidates else np.zeros((N, 3))
    
    for pos in candidates:
        if time.time() > deadline:
            break
        try:
            relaxed_pos = science_chemistry.relax(pos)
            # Calculate energy
            energy = science_chemistry.lj_energy(relaxed_pos)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            continue
            
    return best_pos
