import time

def construct():
    # The maximum size is 31, achieved by the arithmetic progression {226, ..., 256}.
    # This set satisfies the condition that the maximum sum of k-1 elements
    # is strictly less than the minimum sum of k elements for all k.
    # Verification:
    # Min sum of k elements: sum_{i=0}^{k-1} (226+i)
    # Max sum of k-1 elements: sum_{i=0}^{k-2} (256-i)
    # Gap is positive for all k=1..31.
    return list(range(226, 257))
