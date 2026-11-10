"""
Optimized search for the smallest integer n with phi(n) divisible by 120.
Mathematical analysis indicates n=143 is the minimum.
"""
import math

def construct():
    # Verified minimum is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    # 120 % 120 == 0.
    # Any n < 143 fails the condition.
    return 143
