import numpy as np

def equation(x, y, params):
    # Structural change: sum of two 2D Gaussian blobs with 10 parameters.
    # p0: amplitude of blob 1, p1: x-center of blob 1, p2: y-center of blob 1,
    # p3: x-width of blob 1, p4: y-width of blob 1,
    # p5: amplitude of blob 2, p6: x-center of blob 2, p7: y-center of blob 2,
    # p8: x-width of blob 2, p9: y-width of blob 2.
    # Use softplus to keep widths positive and avoid singularities.
    
    p = params
    # Softplus for widths to ensure positivity and smoothness
    w1x = np.log1p(np.exp(p[3]))
    w1y = np.log1p(np.exp(p[4]))
    w2x = np.log1p(np.exp(p[8]))
    w2y = np.log1p(np.exp(p[9]))
    
    # Clip centers to [-1, 1] to keep them in the domain
    c1x = np.tanh(p[1])
    c1y = np.tanh(p[2])
    c2x = np.tanh(p[6])
    c2y = np.tanh(p[7])
    
    # Two Gaussian blobs
    blob1 = p[0] * np.exp(-((x - c1x)**2 / (2 * w1x**2) + (y - c1y)**2 / (2 * w1y**2)))
    blob2 = p[5] * np.exp(-((x - c2x)**2 / (2 * w2x**2) + (y - c2y)**2 / (2 * w2y**2)))
    
    return blob1 + blob2
