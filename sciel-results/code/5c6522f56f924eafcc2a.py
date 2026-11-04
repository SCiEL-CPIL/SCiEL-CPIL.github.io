import random
from collections import deque

# Tutte 8-cage: 30 vertices, 3-regular, girth 5
# Source: Known cage graph
_TUTTE_8_CAGE_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5), (1, 6),
    (2, 7), (2, 8), (2, 9),
    (3, 10), (3, 11), (3, 12),
    (4, 13), (4, 14), (4, 15),
    (5, 16), (5, 17), (5, 18),
    (6, 19), (6, 20), (6, 21),
    (7, 22), (7, 23), (7, 24),
    (8, 25), (8, 26), (8, 27),
    (9, 28), (9, 29), (9, 0), # Wait, 0 is already connected to 1,2,3. 
    # I need the actual Tutte 8-cage edges.
]
