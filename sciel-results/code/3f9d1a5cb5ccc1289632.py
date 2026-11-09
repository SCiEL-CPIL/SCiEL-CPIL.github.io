<<<<<<< SEARCH
def _initial_parallelogram():
    """Parallelogram-shaped grid."""
    points_list = []
    for i in range(6):
        for j in range(9):
            x = i + j // 3
            y = j
            points_list.append((x + 1, y + 1))
    return np.array(points_list, dtype=np.int64)
=======
def _initial_parallelogram():
    """Parallelogram-shaped grid."""
    points_list = []
    for i in range(6):
        for j in range(9):
            x = i + j // 3
            y = j
            points_list.append((x + 1, y + 1))
    return np.array(points_list, dtype=np.int64)

def _initial_grid_6x9_offset():
    """6x9 grid with slight row offsets to create more diagonal lines."""
    points_list = []
    for i in range(6):
        offset = i % 2  # alternate offset
        for j in range(9):
            x = i + 1 + offset
            y = j + 1
            points_list.append((x, y))
    return np.array(points_list, dtype=np.int64)

def _initial_grid_6x9_sheared():
    """6x9 sheared grid: x shifted by y//2."""
    points_list = []
    for i in range(6):
        for j in range(9):
            x = i + j // 3 + 1
            y = j + 1
            points_list.append((x, y))
    return np.array(points_list, dtype=np.int64)

def _initial_3x18_compact():
    """3 columns of 18, but with some horizontal offsets."""
    points_list = []
    for col in range(3):
        for row in range(18):
            x = col + 1 + (row % 2)  # zigzag
            y = row + 1
            points_list.append((x, y))
    return np.array(points_list, dtype=np.int64)
>>>>>>> REPLACE

