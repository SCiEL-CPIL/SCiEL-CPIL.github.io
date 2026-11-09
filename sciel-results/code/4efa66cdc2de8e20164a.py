import numpy as np

def szemeredi_trotter_incidences():
    # 7x8 grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Remove (4, 1) and (4, 8)
    # (4, 1) is index (3, 0) in 0-based (x-1, y-1)
    # (4, 8) is index (3, 7)
    # Flatten index: (x-1)*8 + (y-1)
    idx1 = (4-1)*8 + (1-1) # 24
    idx2 = (4-1)*8 + (8-1) # 31
    
    # Create mask
    mask = np.ones(56, dtype=bool)
    mask[idx1] = False
    mask[idx2] = False
    points = points[mask]
    
    # Lines
    lines = []
    # 7 Verticals: x = k -> 1*x + 0*y - k = 0
    for x in range(1, 8):
        lines.append((1, 0, -x))
        
    # 8 Horizontals: y = k -> 0*x + 1*y - k = 0
    for y in range(1, 9):
        lines.append((0, 1, -y))
        
    # 4 Diagonals Len 7 (Slope 1)
    # y = x + b -> x - y + b = 0
    for b in [0, 1]:
        lines.append((1, -1, b))
        
    # 4 Diagonals Len 7 (Slope -1)
    # x + y = c -> x + y - c = 0
    for c in [8, 9]:
        lines.append((1, 1, -c))
        
    # 4 Diagonals Len 6 (Slope 1)
    for b in [-1, 2]:
        lines.append((1, -1, b))
        
    # 4 Diagonals Len 6 (Slope -1)
    for c in [7, 10]:
        lines.append((1, 1, -c))
        
    # 4 Diagonals Len 5 (Slope 1)
    for b in [-2, 3]:
        lines.append((1, -1, b))
        
    # 4 Diagonals Len 5 (Slope -1)
    for c in [6, 11]:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
