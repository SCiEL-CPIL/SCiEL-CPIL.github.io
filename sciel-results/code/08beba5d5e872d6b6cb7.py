"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Perimeter = 4 => w + h = 2, so h = 2 - w
    # We'll optimize over w and circle positions/radii
    
    # Initial guess: try a few aspect ratios
    best_result = None
    best_score = -1
    
    # Try different width values
    for w in np.linspace(0.8, 1.2, 15):
        h = 2.0 - w
        
        # Initialize circles in a hexagonal-like pattern
        # Estimate how many fit per row
        # For now, place them in a grid and then relax
        cols = int(np.sqrt(N * w / h)) if h > 0 else N
        cols = max(1, min(cols, N))
        rows = (N + cols - 1) // cols
        
        # Initial positions: grid
        positions = []
        for i in range(cols):
            for j in range(rows):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                positions.append([x, y])
        positions = np.array(positions[:N])
        
        # Initial radius: min distance to boundary
        radii = np.zeros(N)
        for i in range(N):
            radii[i] = min(positions[i, 0], w - positions[i, 0], 
                          positions[i, 1], h - positions[i, 1])
        
        # Iterative relaxation
        for iteration in range(2000):
            # Try to increase each radius slightly
            for i in range(N):
                # Max possible radius for circle i given boundaries
                max_r = min(positions[i, 0], w - positions[i, 0],
                           positions[i, 1], h - positions[i, 1])
                
                # Check overlap with others
                min_allowed = max_r
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    # To not overlap: r_i + r_j <= dist
                    # So r_i <= dist - r_j
                    allowed = dist - radii[j]
                    if allowed < min_allowed:
                        min_allowed = allowed
                
                min_allowed = max(0, min_allowed)
                
                # Try to set radius to min_allowed or increase toward it
                target_r = min_allowed
                if target_r > radii[i]:
                    # Move position to center more if possible, or just increase
                    radii[i] = target_r
                else:
                    radii[i] = max(radii[i] * 0.99, target_r)
            
            # Push overlapping circles apart
            for i in range(N):
                for j in range(i+1, N):
                    diff = positions[j] - positions[i]
                    dist = np.sqrt(np.sum(diff**2))
                    required = radii[i] + radii[j]
                    if dist < required and dist > 1e-10:
                        # Push apart
                        push = (required - dist) / 2.0
                        direction = diff / dist
                        positions[i] -= push * direction
                        positions[j] += push * direction
                        # Clamp to boundaries
                        positions[i, 0] = np.clip(positions[i, 0], 0, w)
                        positions[i, 1] = np.clip(positions[i, 1], 0, h)
                        positions[j, 0] = np.clip(positions[j, 0], 0, w)
                        positions[j, 1] = np.clip(positions[j, 1], 0, h)
            
            # Update radii after position changes
            for i in range(N):
                max_r = min(positions[i, 0], w - positions[i, 0],
                           positions[i, 1], h - positions[i, 1])
                min_allowed = max_r
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    allowed = dist - radii[j]
                    if allowed < min_allowed:
                        min_allowed = allowed
                radii[i] = max(0, min(min_allowed, max_r))
        
        # Final verification and cleanup
        # Ensure all circles are within bounds and non-overlapping
        valid = True
        for i in range(N):
            if positions[i, 0] < -1e-9 or positions[i, 0] > w + 1e-9 or \
               positions[i, 1] < -1e-9 or positions[i, 1] > h + 1e-9 or \
               radii[i] < -1e-9:
                valid = False
                break
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if dist < radii[i] + radii[j] - 1e-7:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii])
    
    if best_result is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
