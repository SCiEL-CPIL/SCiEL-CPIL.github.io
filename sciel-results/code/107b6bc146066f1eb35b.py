"""
Optimal OneMax solution: Construct the string of all 1s.
"""

N = 100

def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The global optimum for OneMax is a string consisting entirely of '1's.
    This function directly returns that string.
    """
    return "1" * N
