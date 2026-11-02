import numpy as np

N = 50

def construct():
    # Triangular pulse family: h(x) = max(0, 1 - |x|/width)
    # Narrower width concentrates L2 mass more, potentially improving ratio
    # Try different widths to find optimal shape
    x = np.linspace(-1, 1, N)
    
    # Use a narrower triangular pulse: support on [-w, w] within [-1, 1]
    # The shape is (1 - |x|/w) for |x| <= w, 0 otherwise
    # Try w = 0.5 (half the original width)
    w = 0.5
    h_shape = np.maximum(0, 1 - np.abs(x) / w)
    
    # Normalize so that max(h_shape) = 1 (it already is)
    # Scale down to maintain the same L1 norm behavior as before
    # Original triangular had L1 ~ sum(1-|x|) ~ N/2 = 25
    # New one has L1 ~ sum(max(0, 1-|x|/0.5)) over |x|<=0.5
    # This is approximately (0.5 * 0.5) * N * 2 / 2 = ... let's just compute
    # Actually, let's keep the same scale factor approach
    scale = 1e-6
    return (h_shape * scale).tolist()
