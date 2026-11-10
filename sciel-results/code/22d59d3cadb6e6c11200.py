import numpy as np

def szemeredi_trotter_incidences():
    # Grid 9x6
    # Points: x in 1..9, y in 1..6
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 6 horizontal lines y = b -> 0*x + 1*y - b = 0
    for b in range(1, 7):
        lines.append((0, 1, -b))
        
    # 9 vertical lines x = a -> 1*x + 0*y - a = 0
    for a in range(1, 10):
        lines.append((1, 0, -a))
        
    # 4 lines y = x + b -> 1*x - 1*y - b = 0
    # b = 0, -1, -2, -3
    for b in [0, -1, -2, -3]:
        lines.append((1, -1, -b))
        
    # 4 lines y = -x + b -> 1*x + 1*y - b = 0
    # b = 10, 9, 8, 7
    for b in [10, 9, 8, 7]:
        lines.append((1, 1, -b))
        
    # 4 more lines to reach 27
    # y = x - 4 -> 1*x - 1*y + 4 = 0
    lines.append((1, -1, 4))
    # y = -x + 6 -> 1*x + 1*y - 6 = 0
    lines.append((1, 1, -6))
    # y = x - 5 -> 1*x - 1*y + 5 = 0
    lines.append((1, -1, 5))
    # y = -x + 5 -> 1*x + 1*y - 5 = 0
    lines.append((1, 1, -5))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
