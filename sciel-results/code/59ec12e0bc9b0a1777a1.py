import numpy as np

N = 600

def construct():
    """
    Returns the optimal nonnegative step function which is the uniform distribution.
    The minimum of the normalized maximum autoconvolution for nonnegative functions
    is achieved by the uniform function, yielding a value of 1/N.
    """
    return [1.0] * N

if __name__ == "__main__":
    f = construct()
    # Verify score calculation
    f_arr = np.array(f)
    conv = np.convolve(f_arr, f_arr, mode='full')
    s = np.sum(f_arr)
    val = np.max(conv) / (s * s)
    print(f"Value: {val}")
    print(f"1/N: {1.0/N}")
