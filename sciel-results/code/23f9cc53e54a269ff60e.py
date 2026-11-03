import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate x coordinates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * (0.5 / N)
    
    # Triangle function
    f = np.maximum(0.0, 1.0 - 4.0 * np.abs(x))
    
    # Optional: Try a few variations if time permits
    # But triangle is a strong candidate.
    
    return f.tolist()
# EVOLVE-BLOCK-END
