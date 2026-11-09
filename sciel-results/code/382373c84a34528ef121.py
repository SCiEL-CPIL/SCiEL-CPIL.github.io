import numpy as np


def szemeredi_trotter_incidences():
    """
    Construct 54 points and 27 lines to maximize incidences using a 3x18 grid.
    
    Points: x in {1,2,3}, y in {1..18} -> 54 points
    Lines:
      - 18 horizontal: y = j (j=1..18), each with 3 points -> 54 incidences
      - 3 vertical: x = i (i=1,2,3), each with 18 points -> 54 incidences
      - 12 slope -1: x + y = c for c in {4..15}, each with 3 points -> 36 incidences
      - 12 slope +1: y - x = k for k in {0..11}, each with 3 points -> 36 incidences
    Total: 54 + 54 + 36 + 36 = 180 incidences
    """
    xs = np.array([1, 2, 3], dtype=np.int64)
    ys = np.arange(1, 19, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 18 Horizontal lines: y = j => 0x + 1y - j = 0
    for j in range(1, 19):
        lines.append((0, 1, -j))
    
    # 3 Vertical lines: x = i => 1x + 0y - i = 0
    for i in range(1, 4):
        lines.append((1, 0, -i))
    
    # 12 Slope -1 lines: x + y = c => 1x + 1y - c = 0
    # c in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    for c in range(4, 16):
        lines.append((1, 1, -c))
    
    # 12 Slope +1 lines: y - x = k => -1x + 1y - k = 0  (or x - y + k = 0 with sign flip)
    # Use: -x + y - k = 0 => A=-1, B=1, C=-k
    # But to keep first nonzero positive, use: x - y + k = 0? No: y = x + k => x - y + k = 0
    # Wait: y = x + k => x - y + k = 0. Check: x=1,y=1,k=0: 1-1+0=0 ✓
    # So A=1, B=-1, C=k
    for k in range(0, 12):
        lines.append((1, -1, k))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
