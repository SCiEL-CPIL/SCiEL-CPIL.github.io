"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _valid_pair(a, b):
    # Returns True if (a*b) % (a+b) != 0
    return (a * b) % (a + b) != 0

def _is_valid_set(S):
    # Check all distinct pairs
    lst = list(S)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lst[i], lst[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (limit is 90s)
    
    # Start with the known good seed: all odd numbers in [1, 256]
    # This has 128 elements.
    # Let's verify it's valid quickly.
    # For odd a, b: a+b is even, ab is odd. Odd % Even != 0 is always true.
    # So all odds are valid.
    current_best = list(range(1, 257, 2))
    best_size = len(current_best)
    
    # Strategy:
    # 1. Try to add even numbers to the set of odds.
    # 2. If adding an even number fails, try to remove some odds to make room for more evens?
    #    Or just try to add evens one by one, and if conflict, skip.
    #    Actually, a greedy approach:
    #    Start with odds.
    #    For each even number e in [1, 256]:
    #       Check if e is compatible with ALL current members.
    #       If yes, add it.
    #       If no, skip it.
    #    This might miss combinations where removing one odd allows adding two evens.
    #    
    #    Let's analyze the constraint for even e and odd o:
    #    (e * o) % (e + o) != 0
    #    Let e = 2k. o is odd.
    #    (2k * o) % (2k + o) != 0
    #    2ko = q(2k+o) + r.
    #    Note that 2k+o is odd (even+odd).
    #    2ko is even.
    #    If 2k+o divides 2ko, then 2k+o must be odd (which it is).
    #    
    #    Let's try a local search / simulated annealing style approach or just smart greedy.
    #    
    #    Alternative: Start with all odds.
    #    Identify "conflicting" even numbers.
    #    An even number e is incompatible with an odd number o if (e*o) % (e+o) == 0.
    #    
    #    Let's compute the set of odds that conflict with each even number.
    #    If an even number conflicts with many odds, it's hard to include.
    #    
    #    Let's try to maximize the set by starting with odds and greedily adding evens.
    #    If that doesn't improve, try a random restart with a different initial set?
    #    The odds set is likely a strong local optimum.
    #    
    #    Let's check if we can add ANY evens to the full set of odds.
    #    If we take all odds, can we add any even?
    #    Let's test e=2.
    #    Conflict with o if (2*o) % (2+o) == 0.
    #    2o = k(2+o) => 2o - ko = 2k => o(2-k) = 2k => o = 2k/(2-k).
    #    If k=1, o = 2/1 = 2 (not odd).
    #    If k=0, o=0.
    #    If k=3, o = 6/-1 = -6.
    #    It seems 2 is compatible with all odds?
    #    Let's check: (2*3)%(2+3) = 6%5=1 !=0. (2*5)%7=3!=0.
    #    Wait, (2*o) % (2+o).
    #    Let o=1. 2%3=2.
    #    Let o=3. 6%5=1.
    #    Let o=5. 10%7=3.
    #    Let o=7. 14%9=5.
    #    Let o=9. 18%11=7.
    #    It seems 2 is compatible with all odds.
    #    So we can add 2.
    #    
    #    Let's check e=4.
    #    (4*o) % (4+o) != 0.
    #    o=1: 4%5=4.
    #    o=3: 12%7=5.
    #    o=5: 20%9=2.
    #    o=7: 28%11=6.
    #    o=9: 36%13=10.
    #    o=11: 44%15=14.
    #    o=13: 52%17=1.
    #    o=15: 60%19=3.
    #    o=17: 68%21=5.
    #    o=19: 76%23=7.
    #    o=21: 84%25=9.
    #    o=23: 92%27=11.
    #    o=25: 100%29=13.
    #    o=27: 108%31=15.
    #    o=29: 116%33=17.
    #    o=31: 124%35=19.
    #    o=33: 132%37=21.
    #    o=35: 140%39=23.
    #    o=37: 148%41=25.
    #    o=39: 156%43=27.
    #    o=41: 164%45=29.
    #    o=43: 172%47=31.
    #    o=45: 180%49=33.
    #    o=47: 188%51=35.
    #    o=49: 196%53=37.
    #    o=51: 204%55=39.
    #    o=53: 212%57=41.
    #    o=55: 220%59=43.
    #    o=57: 228%61=45.
    #    o=59: 236%63=47.
    #    o=61: 244%65=49.
    #    o=63: 252%67=51.
    #    o=65: 260%69=53.
    #    o=67: 268%71=55.
    #    o=69: 276%73=57.
    #    o=71: 284%75=59.
    #    o=73: 292%77=61.
    #    o=75: 300%79=63.
    #    o=77: 308%81=65.
    #    o=79: 316%83=67.
    #    o=81: 324%85=69.
    #    o=83: 332%87=71.
    #    o=85: 340%89=73.
    #    o=87: 348%91=75.
    #    o=89: 356%93=77.
    #    o=91: 364%95=79.
    #    o=93: 372%97=81.
    #    o=95: 380%99=83.
    #    o=97: 388%101=85.
    #    o=99: 396%103=87.
    #    o=101: 404%105=89.
    #    o=103: 412%107=91.
    #    o=105: 420%109=93.
    #    o=107: 428%111=95.
    #    o=109: 436%113=97.
    #    o=111: 444%115=99.
    #    o=113: 452%117=101.
    #    o=115: 460%119=103.
    #    o=117: 468%121=105.
    #    o=119: 476%123=107.
    #    o=121: 484%125=109.
    #    o=123: 492%127=111.
    #    o=125: 500%129=113.
    #    o=127: 508%131=115.
    #    o=129: 516%133=117.
    #    o=131: 524%135=119.
    #    o=133: 532%137=121.
    #    o=135: 540%139=123.
    #    o=137: 548%141=125.
    #    o=139: 556%143=127.
    #    o=141: 564%145=129.
    #    o=143: 572%147=131.
    #    o=145: 580%149=133.
    #    o=147: 588%151=135.
    #    o=149: 596%153=137.
    #    o=151: 604%155=139.
    #    o=153: 612%157=141.
    #    o=155: 620%159=143.
    #    o=157: 628%161=145.
    #    o=159: 636%163=147.
    #    o=161: 644%165=149.
    #    o=163: 652%167=151.
    #    o=165: 660%169=153.
    #    o=167: 668%171=155.
    #    o=169: 676%173=157.
    #    o=171: 684%175=159.
    #    o=173: 692%177=161.
    #    o=175: 700%179=163.
    #    o=177: 708%181=165.
    #    o=179: 716%183=167.
    #    o=181: 724%185=169.
    #    o=183: 732%187=171.
    #    o=185: 740%189=173.
    #    o=187: 748%191=175.
    #    o=189: 756%193=177.
    #    o=191: 764%195=179.
    #    o=193: 772%197=181.
    #    o=195: 780%199=183.
    #    o=197: 788%201=185.
    #    o=199: 796%203=187.
    #    o=201: 804%205=189.
    #    o=203: 812%207=191.
    #    o=205: 820%209=193.
    #    o=207: 828%211=195.
    #    o=209: 836%213=197.
    #    o=211: 844%215=199.
    #    o=213: 852%217=201.
    #    o=215: 860%219=203.
    #    o=217: 868%221=205.
    #    o=219: 876%223=207.
    #    o=221: 884%225=209.
    #    o=223: 892%227=211.
    #    o=225: 900%229=213.
    #    o=227: 908%231=215.
    #    o=229: 916%233=217.
    #    o=231: 924%235=219.
    #    o=233: 932%237=221.
    #    o=235: 940%239=223.
    #    o=237: 948%241=225.
    #    o=239: 956%243=227.
    #    o=241: 964%245=229.
    #    o=243: 972%247=231.
    #    o=245: 980%249=233.
    #    o=247: 988%251=235.
    #    o=249: 996%253=237.
    #    o=251: 1004%255=239.
    #    o=253: 1012%257=241.
    #    o=255: 1020%259=243.
    #    
    #    It seems 4 is also compatible with all odds.
    #    
    #    Let's check e=6.
    #    (6*o) % (6+o) != 0.
    #    o=1: 6%7=6.
    #    o=3: 18%9=0. CONFLICT!
    #    So 6 conflicts with 3.
    #    
    #    So we can add 2 and 4 to the set of odds.
    #    Can we add more evens?
    #    Let's try to add evens greedily to the set of odds + {2, 4}.
    #    
    #    Let's write a function to greedily add evens.
    
    def greedy_add_evens(initial_set):
        S = set(initial_set)
        # Sort evens
        evens = [i for i in range(2, 257, 2) if i not in S]
        # Maybe try adding evens in increasing order? Or random?
        # Let's try increasing first.
        for e in evens:
            compatible = True
            for o in S:
                if (e * o) % (e + o) == 0:
                    compatible = False
                    break
            if compatible:
                S.add(e)
        return S
    
    # Let's run this.
    S_greedy = greedy_add_evens(current_best)
    if len(S_greedy) > best_size:
        current_best = list(S_greedy)
        best_size = len(current_best)
        
    # Now, can we do better?
    # Maybe we can remove some odds to add more evens?
    # The conflict for even e and odd o is (e*o) % (e+o) == 0.
    # This is equivalent to (e+o) | (e*o).
    # Since gcd(e, e+o) = gcd(e, o) = 1 (if e even, o odd),
    # e+o must divide o? No.
    # Let e=2k, o=odd.
    # (2k+o) | 2ko.
    # Since gcd(2k+o, o) = gcd(2k, o) = 1 (o odd, 2k even),
    # 2k+o must divide 2k.
    # But 2k+o > 2k for o>0.
    # So 2k+o cannot divide 2k unless 2k=0.
    # Wait.
    # gcd(2k+o, o) = gcd(2k, o).
    # If o is odd and 2k is even, gcd(2k, o) can be > 1.
    # Example: e=6 (k=3), o=3. gcd(6,3)=3.
    # 2k+o = 9. 2ko = 18. 9|18. Yes.
    # So the condition is:
    # Let d = gcd(e, o).
    # Then e/d and o/d are coprime.
    # e+o = d(e/d + o/d).
    # e*o = d^2 (e/d)(o/d).
    # (e+o) | (e*o)  <=>  d(e/d + o/d) | d^2 (e/d)(o/d)
    # <=>  (e/d + o/d) | d (e/d)(o/d)
    # Since gcd(e/d + o/d, e/d) = gcd(o/d, e/d) = 1,
    # and gcd(e/d + o/d, o/d) = gcd(e/d, o/d) = 1,
    # We have gcd(e/d + o/d, (e/d)(o/d)) = 1.
    # Therefore, (e/d + o/d) must divide d.
    # So the conflict condition is:
    # Let e, o be the pair.
    # d = gcd(e, o).
    # e' = e/d, o' = o/d.
    # (e' + o') divides d.
    
    # This is a very useful characterization!
    # Conflict if (e/gcd(e,o) + o/gcd(e,o)) divides gcd(e,o).
    
    # Let's use this to optimize the search.
    # We want to maximize |S|.
    # Start with all odds.
    # Try to add evens.
    # If an even e conflicts with an odd o, we can either drop o or drop e.
    # Dropping o costs 1, dropping e costs 1.
    # But dropping o might allow adding other evens that conflict with o.
    # So it's a weighted set cover / independent set problem on a graph.
    # Nodes: 1..256.
    # Edge between a,b if (a*b)%(a+b)==0.
    # Find Max Independent Set.
    
    # Since we have a time limit, we can use a heuristic.
    # 1. Start with all odds.
    # 2. For each even, check conflicts with current set.
    #    If no conflict, add it.
    #    If conflict, check if removing the conflicting odds allows adding this even AND some other evens?
    #    This is complex.
    
    # Let's stick to the greedy addition of evens to odds, and then try some random restarts with different initial sets?
    # Or just improve the greedy by trying different orders of evens?
    
    # Let's try a local search:
    # Start with S_greedy.
    # Try to add any element not in S.
    # If it conflicts, try to remove one of the conflicting elements.
    # If the size increases, accept.
    
    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add each element not in S
            for x in range(1, 257):
                if x in S:
                    continue
                # Check conflicts
                conflicts = []
                for y in S:
                    if (x * y) % (x + y) == 0:
                        conflicts.append(y)
                if not conflicts:
                    S.add(x)
                    improved = True
                else:
                    # Try to remove one conflict and add x
                    # Only worth it if we can add x and maybe others?
                    # For now, just try removing one conflict.
                    for c in conflicts:
                        new_S = S - {c}
                        new_S.add(x)
                        # Check if new_S is valid (it should be, since we only removed one and added one)
                        # But we need to check if x conflicts with others in new_S?
                        # x only conflicts with c in S (and maybe others? No, we iterated all y in S).
                        # So x is compatible with S - {c}.
                        # And c was compatible with S - {c} (since S was valid).
                        # So new_S is valid.
                        # Size is same. No gain.
                        # So this step doesn't help increase size.
                        pass
        return S
    
    # Local search as defined above doesn't increase size.
    # We need to find a way to increase size.
    # This requires removing k elements and adding m > k elements.
    
    # Let's try a different approach:
    # Randomly generate a set, then greedily add elements.
    # Or start with a random subset of odds and evens, then greedily add.
    
    # Given the time, let's just output the result of greedy_add_evens.
    # It likely adds 2 and 4. Maybe more?
    # Let's check 8.
    # e=8. o=odd.
    # d = gcd(8, o). Since o is odd, d is odd.
    # e' = 8/d, o' = o/d.
    # Condition: (8/d + o/d) divides d.
    # 8/d + o/d = (8+o)/d.
    # So (8+o)/d divides d => d^2 divides 8+o.
    # Since d divides 8, d can be 1.
    # If d=1, 1 divides 8+o. Always true.
    # So 8 is compatible with all odds?
    # Let's check o=1. 8%9=8.
    # o=3. 24%11=2.
    # o=5. 40%13=1.
    # o=7. 56%15=11.
    # o=9. 72%17=4.
    # o=11. 88%19=12.
    # o=13. 104%21=20.
    # o=15. 120%23=5.
    # o=17. 136%25=11.
    # o=19. 152%27=17.
    # o=21. 168%29=23.
    # o=23. 184%31=28.
    # o=25. 200%33=2.
    # o=27. 216%35=1.
    # o=29. 232%37=10.
    # o=31. 248%39=14.
    # o=33. 264%41=38.
    # o=35. 280%43=25.
    # o=37. 296%45=26.
    # o=39. 312%47=31.
    # o=41. 328%49=34.
    # o=43. 344%51=38.
    # o=45. 360%53=38.
    # o=47. 376%55=26.
    # o=49. 392%57=37.
    # o=51. 408%59=38.
    # o=53. 424%61=58.
    # o=55. 440%63=50.
    # o=57. 456%65=51.
    # o=59. 472%67=55.
    # o=61. 488%69=58.
    # o=63. 504%71=62.
    # o=65. 520%73=60.
    # o=67. 536%75=61.
    # o=69. 552%77=68.
    # o=71. 568%79=68.
    # o=73. 584%81=71.
    # o=75. 600%83=71.
    # o=77. 616%85=71.
    # o=79. 632%87=77.
    # o=81. 648%89=73.
    # o=83. 664%91=82.
    # o=85. 680%93=82.
    # o=87. 696%95=81.
    # o=89. 712%97=88.
    # o=91. 728%99=89.
    # o=93. 744%101=92.
    # o=95. 760%103=91.
    # o=97. 776%105=91.
    # o=99. 792%107=93.
    # o=101. 808%109=97.
    # o=103. 824%111=101.
    # o=105. 840%113=104.
    # o=107. 856%115=101.
    # o=109. 872%117=104.
    # o=111. 888%119=108.
    # o=113. 904%121=100.
    # o=115. 920%123=104.
    # o=117. 936%125=111.
    # o=119. 952%127=106.
    # o=121. 968%129=110.
    # o=123. 984%131=122.
    # o=125. 1000%133=117.
    # o=127. 1016%135=121.
    # o=129. 1032%137=116.
    # o=131. 1048%139=127.
    # o=133. 1064%141=121.
    # o=135. 1080%143=125.
    # o=137. 1096%145=131.
    # o=139. 1112%147=136.
    # o=141. 1128%149=131.
    # o=143. 1144%151=132.
    # o=145. 1160%153=131.
    # o=147. 1176%155=136.
    # o=149. 1192%157=141.
    # o=151. 1208%159=140.
    # o=153. 1224%161=143.
    # o=155. 1240%163=147.
    # o=157. 1256%165=151.
    # o=159. 1272%167=155.
    # o=161. 1288%169=159.
    # o=163. 1304%171=163.
    # o=165. 1320%173=167.
    # o=167. 1336%175=161.
    # o=169. 1352%177=170.
    # o=171. 1368%179=170.
    # o=173. 1384%181=172.
    # o=175. 1400%183=172.
    # o=177. 1416%185=176.
    # o=179. 1432%187=179.
    # o=181. 1448%189=182.
    # o=183. 1464%191=182.
    # o=185. 1480%193=184.
    # o=187. 1496%195=181.
    # o=189. 1512%197=188.
    # o=191. 1528%199=190.
    # o=193. 1544%201=191.
    # o=195. 1560%203=192.
    # o=197. 1576%205=196.
    # o=199. 1592%207=197.
    # o=201. 1608%209=199.
    # o=203. 1624%211=202.
    # o=205. 1640%213=204.
    # o=207. 1656%215=206.
    # o=209. 1672%217=208.
    # o=211. 1688%219=210.
    # o=213. 1704%221=212.
    # o=215. 1720%223=214.
    # o=217. 1736%225=216.
    # o=219. 1752%227=218.
    # o=221. 1768%229=220.
    # o=223. 1784%231=222.
    # o=225. 1800%233=224.
    # o=227. 1816%235=226.
    # o=229. 1832%237=228.
    # o=231. 1848%239=230.
    # o=233. 1864%241=232.
    # o=235. 1880%243=234.
    # o=237. 1896%245=236.
    # o=239. 1912%247=238.
    # o=241. 1928%249=240.
    # o=243. 1944%251=242.
    # o=245. 1960%253=244.
    # o=247. 1976%255=246.
    # o=249. 1992%257=248.
    # o=251. 2008%259=250.
    # o=253. 2024%261=252.
    # o=255. 2040%263=254.
    #
    # It seems 8 is also compatible with all odds.
    #
    # Let's check 10.
    # e=10. o=odd.
    # d = gcd(10, o). d can be 1 or 5.
    # If d=1, (10+o) divides 1. Impossible for o>0.
    # If d=5, o is multiple of 5. o=5, 15, 25...
    # e'=2, o'=1, 3, 5...
    # Condition: (2+o') divides 5.
    # o'=1 => 3 divides 5? No.
    # o'=3 => 5 divides 5? Yes. Conflict with o=15.
    # o'=5 => 7 divides 5? No.
    # o'=7 => 9 divides 5? No.
    # So 10 conflicts with 15.
    
    # So the greedy algorithm will likely add 2, 4, 8.
    # Let's see if it adds 10.
    # If S contains 15, 10 will be skipped.
    # 15 is in the initial odds set.
    # So 10 is skipped.
    
    # Can we do better?
    # If we remove 15, we can add 10. Size change: 0.
    # But maybe removing 15 allows adding 10 AND another even?
    
    # Let's just run the greedy algorithm and return the result.
    # It's better than the seed (128) if it adds any evens.
    
    return sorted(current_best)

# EVOLVE-BLOCK-END
