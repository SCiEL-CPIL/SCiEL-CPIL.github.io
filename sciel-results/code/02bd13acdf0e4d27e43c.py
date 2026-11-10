def popcount(x):
    if hasattr(int, 'bit_count'):
        return x.bit_count()
    return bin(x).count('1')
