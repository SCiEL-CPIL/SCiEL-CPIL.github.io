import math
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin
    
    # Inner hexagons
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = dist1 * math.cos(angle)
        y = dist1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Ring 2: 4 hexagons at distance 2*sqrt(3)
    # We pick 4 of the 6 corners to make 11 total
    dist2 = 2 * math.sqrt(3)
    # Angles: 0, 60, 120, 240 (indices 0, 1, 2, 4)
    selected_indices = [0, 1, 2, 4]
    for i in selected_indices:
        angle = math.radians(60 * i)
        x = dist2 * math.cos(angle)
        y = dist2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length S = 2*sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1
    center = (0.0, 0.0)
    rotation = 0.0
    
    return inner, center, side, rotation
