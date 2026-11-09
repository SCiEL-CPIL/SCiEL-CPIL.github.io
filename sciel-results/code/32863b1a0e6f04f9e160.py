import numpy as np

def szemeredi_trotter_incidences():
    # 6 x 9 grid: x in 1..6, y in 1..9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 6 vertical lines: x = i
    for i in range(1, 7):
        lines.append((1, 0, -i))
        
    # 9 horizontal lines: y = j
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 7 slope-1 lines: y = x + c -> x - y + c = 0
    # c=0: 6 pts, c=1: 6, c=2: 6, c=3: 6, c=-1: 5, c=4: 5, c=-2: 4
    lines.append((1, -1, 0))   # c=0, 6 pts
    lines.append((1, -1, 1))   # c=1, 6 pts
    lines.append((1, -1, 2))   # c=2, 6 pts
    lines.append((1, -1, 3))   # c=3, 6 pts
    lines.append((1, -1, -1))  # c=-1, 5 pts
    lines.append((1, -1, 4))   # c=4, 5 pts
    lines.append((1, -1, -2))  # c=-2, 4 pts
    
    # 5 slope-2 lines: y = 2x + c -> 2x - y + c = 0
    # c=-1: 4 pts, c=0: 4, c=1: 4, c=-2: 4, c=2: 4
    lines.append((2, -1, -1))  # c=-1, 4 pts
    lines.append((2, -1, 0))   # c=0, 4 pts
    lines.append((2, -1, 1))   # c=1, 4 pts
    lines.append((2, -1, -2))  # c=-2, 4 pts
    lines.append((2, -1, 2))   # c=2, 4 pts
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
