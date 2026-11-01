import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # The recursive construction yields 33, which is optimal.
        # We can just let the general loop handle it or hardcode.
        # General loop:
        # S_3 = 123121321
        # Perms of 1,2,3 in order: 123, 231, 312, 132, 321, 213
        # Wait, let's trace S_3 from S_2.
        # S_2 = 121. Perms: 12, 21.
        # Block 1: 12 + 3 + 12 = 12312
        # Block 2: 21 + 3 + 21 = 21321
        # Overlap 12312 and 21321? 
        # Suffix 12, Prefix 21. No overlap?
        # 12312 + 21321 = 1231221321. Length 10.
        # But optimal S_3 is 123121321 (Length 9).
        # Why the difference?
        # In 123121321:
        # 123
        # 231
        # 312
        # 132
        # 321
        # 213
        # The standard recursive construction I described above (P + n + P) might not be the one generating the optimal S_3?
        # Let's check the seed code logic.
        # Seed code:
        # word = "1"
        # m=2:
        #   perms from "1": none? 
        #   Wait. 
        #   for i in range(len(word) - (m-1) + 1):
        #   if m=2, m-1=1.
        #   word="1". len=1. range(1-1+1)=range(1). i=0.
        #   window = word[0:1] = "1".
        #   len(set("1"))==1. unique.
        #   perms = ["1"].
        #   new_word = ""
        #   for p in ["1"]:
        #     block = "1" + "2" + "1" = "121".
        #     overlap with "" is 0.
        #     new_word = "121".
        #   word = "121".
        # m=3:
        #   perms from "121" of len 2.
        #   i=0: "12". unique. add "12".
        #   i=1: "21". unique. add "21".
        #   perms = ["12", "21"].
        #   new_word = ""
        #   p="12": block = "12312". new_word="12312".
        #   p="21": block = "21321".
        #   overlap "12312" suffix vs "21321" prefix.
        #   "2" vs "2"? 
        #   new_word[-1:] = "2". block[:1] = "2". Match.
        #   new_word[-2:] = "12". block[:2] = "21". No.
        #   So overlap 1.
        #   append block[1:] = "1321".
        #   new_word = "12312" + "1321" = "123121321".
        #   Length 9. Correct.
        
        # So the seed code IS the standard recursive construction and it works.
        pass

    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            return word # Return partial? No, must be valid. 
            # But the loop is fast.
            
        # Extract unique (m-1) permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            
            # Find max overlap
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
        
    return word
