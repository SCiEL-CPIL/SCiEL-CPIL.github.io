"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    # The score is seed/value, so we want to minimize n.
    # 
    # Let's verify: is 143 the smallest?
    # phi(n) divisible by 120 means 120 | phi(n).
    # 120 = 2^3 * 3 * 5
    # 
    # For phi(n) to be divisible by 5, n must have a prime factor p where p-1 is divisible by 5,
    # or n must be divisible by 5 (in which case phi(n) includes a factor of 4 from phi(5)=4... wait).
    # Actually, if 5 | n, then 4 | phi(n) but not necessarily 5 | phi(n).
    # If p | n and p is prime, then (p-1) | phi(n) only if the power of p in n is exactly 1? No.
    # phi(n) = n * prod(1 - 1/p) for p|n.
    # 
    # Let's just do a direct search from n=1 upward, computing phi(n) and checking divisibility by 120.
    # We can precompute primes up to sqrt(1000000) = 1000, or just up to a reasonable limit.
    # Actually, for each n, we can factor it quickly.
    # 
    # A simpler approach: iterate n from 1 to 1000000, compute phi(n), check if phi(n) % 120 == 0.
    # Return the first such n.
    # 
    # To make it fast, we can use a sieve-like approach to compute phi for all n up to some limit,
    # or just factor each n individually. Since we're looking for the smallest n, we'll likely find it quickly.
    # 
    # Let's verify small values:
    # n=1: phi(1)=1, 1%120 != 0
    # n=2: phi(2)=1, no
    # n=3: phi(3)=2, no
    # n=4: phi(4)=2, no
    # n=5: phi(5)=4, no
    # n=6: phi(6)=2, no
    # n=7: phi(7)=6, no
    # n=8: phi(8)=4, no
    # n=9: phi(9)=6, no
    # n=10: phi(10)=4, no
    # n=11: phi(11)=10, no
    # n=12: phi(12)=4, no
    # n=13: phi(13)=12, no
    # n=14: phi(14)=6, no
    # n=15: phi(15)=8, no
    # n=16: phi(16)=8, no
    # n=17: phi(17)=16, no
    # n=18: phi(18)=6, no
    # n=19: phi(19)=18, no
    # n=20: phi(20)=8, no
    # n=21: phi(21)=12, no
    # n=22: phi(22)=10, no
    # n=23: phi(23)=22, no
    # n=24: phi(24)=8, no
    # n=25: phi(25)=20, no
    # n=26: phi(26)=12, no
    # n=27: phi(27)=18, no
    # n=28: phi(28)=12, no
    # n=29: phi(29)=28, no
    # n=30: phi(30)=8, no
    # n=31: phi(31)=30, no
    # n=32: phi(32)=16, no
    # n=33: phi(33)=20, no
    # n=34: phi(34)=16, no
    # n=35: phi(35)=24, no
    # n=36: phi(36)=12, no
    # n=37: phi(37)=36, no
    # n=38: phi(38)=18, no
    # n=39: phi(39)=24, no
    # n=40: phi(40)=16, no
    # n=41: phi(41)=40, no
    # n=42: phi(42)=12, no
    # n=43: phi(43)=42, no
    # n=44: phi(44)=20, no
    # n=45: phi(45)=24, no
    # n=46: phi(46)=22, no
    # n=47: phi(47)=46, no
    # n=48: phi(48)=16, no
    # n=49: phi(49)=42, no
    # n=50: phi(50)=20, no
    # n=51: phi(51)=32, no
    # n=52: phi(52)=24, no
    # n=53: phi(53)=52, no
    # n=54: phi(54)=18, no
    # n=55: phi(55)=40, no
    # n=56: phi(56)=24, no
    # n=57: phi(57)=36, no
    # n=58: phi(58)=28, no
    # n=59: phi(59)=58, no
    # n=60: phi(60)=16, no
    # n=61: phi(61)=60, no
    # n=62: phi(62)=30, no
    # n=63: phi(63)=36, no
    # n=64: phi(64)=32, no
    # n=65: phi(65)=48, no
    # n=66: phi(66)=20, no
    # n=67: phi(67)=66, no
    # n=68: phi(68)=32, no
    # n=69: phi(69)=44, no
    # n=70: phi(70)=24, no
    # n=71: phi(71)=70, no
    # n=72: phi(72)=24, no
    # n=73: phi(73)=72, no
    # n=74: phi(74)=36, no
    # n=75: phi(75)=40, no
    # n=76: phi(76)=36, no
    # n=77: phi(77)=60, no
    # n=78: phi(78)=24, no
    # n=79: phi(79)=78, no
    # n=80: phi(80)=32, no
    # n=81: phi(81)=54, no
    # n=82: phi(82)=40, no
    # n=83: phi(83)=82, no
    # n=84: phi(84)=24, no
    # n=85: phi(85)=64, no
    # n=86: phi(86)=42, no
    # n=87: phi(87)=56, no
    # n=88: phi(88)=40, no
    # n=89: phi(89)=88, no
    # n=90: phi(90)=24, no
    # n=91: phi(91)=72, no
    # n=92: phi(92)=44, no
    # n=93: phi(93)=60, no
    # n=94: phi(94)=46, no
    # n=95: phi(95)=72, no
    # n=96: phi(96)=32, no
    # n=97: phi(97)=96, no
    # n=98: phi(98)=42, no
    # n=99: phi(99)=60, no
    # n=100: phi(100)=40, no
    # n=101: phi(101)=100, no
    # n=102: phi(102)=32, no
    # n=103: phi(103)=102, no
    # n=104: phi(104)=48, no
    # n=105: phi(105)=48, no
    # n=106: phi(106)=52, no
    # n=107: phi(107)=106, no
    # n=108: phi(108)=36, no
    # n=109: phi(109)=108, no
    # n=110: phi(110)=40, no
    # n=111: phi(111)=72, no
    # n=112: phi(112)=48, no
    # n=113: phi(113)=112, no
    # n=114: phi(114)=36, no
    # n=115: phi(115)=88, no
    # n=116: phi(116)=56, no
    # n=117: phi(117)=72, no
    # n=118: phi(118)=58, no
    # n=119: phi(119)=96, no
    # n=120: phi(120)=32, no
    # n=121: phi(121)=110, no
    # n=122: phi(122)=60, no
    # n=123: phi(123)=80, no
    # n=124: phi(124)=60, no
    # n=125: phi(125)=100, no
    # n=126: phi(126)=36, no
    # n=127: phi(127)=126, no
    # n=128: phi(128)=64, no
    # n=129: phi(129)=84, no
    # n=130: phi(130)=48, no
    # n=131: phi(131)=130, no
    # n=132: phi(132)=40, no
    # n=133: phi(133)=108, no
    # n=134: phi(134)=66, no
    # n=135: phi(135)=72, no
    # n=136: phi(136)=64, no
    # n=137: phi(137)=136, no
    # n=138: phi(138)=44, no
    # n=139: phi(139)=138, no
    # n=140: phi(140)=48, no
    # n=141: phi(141)=92, no
    # n=142: phi(142)=70, no
    # n=143: phi(143) = phi(11*13) = phi(11)*phi(13) = 10*12 = 120. 120 % 120 == 0. YES!
    
    # So 143 is a candidate. Is there a smaller one?
    # From the list above, none of 1..142 have phi(n) divisible by 120.
    # Let me double-check a few that might be close:
    # n=60: phi(60)=16
    # n=72: phi(72)=24
    # n=84: phi(84)=24
    # n=90: phi(90)=24
    # n=96: phi(96)=32
    # n=108: phi(108)=36
    # n=120: phi(120)=32
    # n=126: phi(126)=36
    # n=132: phi(132)=40
    # n=138: phi(138)=44
    # n=140: phi(140)=48
    # 
    # What about n where phi(n) could be 120, 240, 360, etc.?
    # For phi(n) to be divisible by 120, phi(n) must be at least 120 (for n>1).
    # So we need phi(n) >= 120 and 120 | phi(n).
    # 
    # Let me check if any n < 143 has phi(n) = 120, 240, 360, etc.
    # Actually, I should just trust the computation. Let me write a quick verification in the code.
    
    def phi(n):
        result = n
        p = 2
        m = n
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
            p += 1
        if m > 1:
            result -= result // m
        return result
    
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    
    # Should not reach here, but just in case
    return 143
# EVOLVE-BLOCK-END
