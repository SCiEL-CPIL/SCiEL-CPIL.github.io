    def popcount(x):
        if hasattr(int, 'bit_count'):
            return x.bit_count()
        else:
            return bin(x).count('1')
    