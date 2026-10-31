import numpy as np

N = 600

def construct():
    # Create a triangle wave
    # Indices 0 to N-1
    # Center at (N-1)/2
    center = (N - 1) / 2.0
    # Distance from center
    dist = np.abs(np.arange(N) - center)
    # Max distance is center
    # Height goes from 1 at center to 0 at edges
    heights = 1.0 - (dist / center)
    # Ensure non-negative (floating point errors)
    heights = np.maximum(heights, 0.0)
    return heights.tolist()
