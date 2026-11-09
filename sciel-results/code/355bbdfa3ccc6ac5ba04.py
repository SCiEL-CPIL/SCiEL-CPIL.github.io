import numpy as np
import time
import itertools

def moving_sofa_3d_path() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    # The corridor is:
    # {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    # The grid is x in linspace(-4,0,81), y and z in linspace(0,1,21)
    # Volume = 0.05^3 * (number of 4x4x4 blocks of adjacent grid points that all survive)
    
    # Key insight: We need to maximize the number of 4x4x4 blocks of grid points that survive.
    # A grid point survives if it's in the corridor at the first pose and after each move,
    # and the axis-aligned box bounding its rotation arc lies in the corridor.
    
    # The evaluator prepends (-3,0,0,0,0,0) and appends (0,6,7,90,90,90).
    # So the starting pose is (-3,0,0,0,0,0) and ending pose is (0,6,7,90,90,90).
    
    # Let's think about what the "sofa" is. The task says "point p is placed at R p + (x,y,z)".
    # The grid points are at specific positions. A grid point survives if it's in the corridor
    # at the first pose and after each move.
    
    # Actually, re-reading: "A grid point survives if it is in the corridor at the first pose 
    # and after each move, and the axis-aligned box bounding its rotation arc lies in the corridor."
    
    # I think the "sofa" is a set of points (the grid), and we're moving it through the corridor.
    # The grid is fixed in the sofa's local frame, and we move the sofa through the corridor.
    
    # Wait, let me re-read more carefully. "The block is the grid x in linspace(-4,0,81), y and z in linspace(0,1,21)."
    # "Each step is done as an x move, y move, z move, then a rotation about a single axis from the old to the new orientation."
    # "A grid point survives if it is in the corridor at the first pose and after each move, and the axis-aligned box bounding its rotation arc lies in the corridor."
    
    # I think the grid points are in the sofa's local coordinate frame. The sofa is moved through the corridor.
    # A grid point (in local frame) survives if, when transformed to world frame by each pose in the path,
    # it lies in the corridor, AND during rotations, the bounding box of the arc it traces is in the corridor.
    
    # Actually, I think I'm overcomplicating this. Let me look at the seed program more carefully.
    
    # The seed returns a path of poses. The evaluator checks if grid points survive.
    # Let me just try to improve the seed path by local search.
    
    # First, let's implement a function to evaluate a path.
    def evaluate_path(path):
        """Evaluate how many 4x4x4 blocks survive for a given path."""
        # Grid
        x_grid = np.linspace(-4, 0, 81)
        y_grid = np.linspace(0, 1, 21)
        z_grid = np.linspace(0, 1, 21)
        
        # Full grid of points
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        
        # The evaluator prepends (-3,0,0,0,0,0) and appends (0,6,7,90,90,90)
        full_path = np.vstack([[-3, 0, 0, 0, 0, 0], path, [0, 6, 7, 90, 90, 90]])
        
        # Insert linearly interpolated poses so each step moves at most 0.05 and changes each angle by at most 2
        # For now, let's just use the given poses directly (the evaluator will interpolate)
        # But we need to simulate what the evaluator does.
        
        # Actually, for evaluation, let's just check if grid points are in the corridor at each pose.
        # The rotation arc bounding box check is complex, so let's approximate.
        
        # For each pose, check if each grid point (transformed to world) is in the corridor.
        # A grid point survives if it's in the corridor at ALL poses.
        
        def in_corridor(x, y, z):
            # {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
            return ((x <= 0) and (0 <= y <= 1) and (0 <= z <= 1)) or \
                   ((0 <= x <= 1) and (0 <= y <= 5) and (0 <= z <= 1)) or \
                   ((0 <= x <= 1) and (5 <= y <= 6) and (z >= 0))
        
        def transform_point(px, py, pz, x, y, z, yaw, pitch, roll):
            # R = Rx(roll) Ry(pitch) Rz(yaw)
            # p_world = R * p_local + (x, y, z)
            yaw_rad = np.radians(yaw)
            pitch_rad = np.radians(pitch)
            roll_rad = np.radians(roll)
            
            cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
            cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
            cr, sr = np.cos(roll_rad), np.sin(roll_rad)
            
            # Rz(yaw)
            x1 = cy * px - sy * py
            y1 = sy * px + cy * py
            z1 = pz
            
            # Ry(pitch)
            x2 = cp * x1 + sp * z1
            y2 = y1
            z2 = -sp * x1 + cp * z1
            
            # Rx(roll)
            x3 = x2
            y3 = cr * y2 - sr * z2
            z3 = sr * y2 + cr * z2
            
            return x3 + x, y3 + y, z3 + z
        
        # Check survival for each grid point
        # For efficiency, we'll check all poses for each grid point
        # But this is O(n_points * n_poses) which could be slow
        
        # Let's use vectorized operations
        n_points = len(X)
        n_poses = len(full_path)
        
        # For each pose, compute where each grid point ends up
        # This is memory intensive, so let's do it in chunks
        
        # Actually, let's just check a subset of grid points for speed
        # Or better, let's check all but be efficient
        
        # For the starting pose (-3,0,0,0,0,0):
        # The grid points are at (-4..0, 0..1, 0..1) in local frame
        # With identity rotation and translation (-3,0,0), they go to (-7..-3, 0..1, 0..1)
        # The corridor has {x<=0, 0<=y<=1, 0<=z<=1}, so all these points are in the corridor.
        
        # For the ending pose (0,6,7,90,90,90):
        # R = Rx(90) Ry(90) Rz(90)
        # Let's compute: Rz(90) maps (x,y,z) -> (-y, x, z)
        # Ry(90) maps (x,y,z) -> (z, y, -x)
        # Rx(90) maps (x,y,z) -> (x, -z, y)
        # Combined: (x,y,z) -> Rz: (-y,x,z) -> Ry: (z, x, y) -> Rx: (z, -y, x)
        # So (px,py,pz) -> (pz, -py, px) + (0,6,7) = (pz, 6-py, px+7)
        # Grid points: px in [-4,0], py in [0,1], pz in [0,1]
        # World: x = pz in [0,1], y = 6-py in [5,6], z = px+7 in [3,7]
        # Corridor: {0<=x<=1, 5<=y<=6, z>=0} - yes, all in corridor.
        
        # So the question is: what path keeps all grid points in the corridor?
        
        # Let's just use the seed path and see if we can improve it.
        # For now, return the seed path.
        return path
    
    # Let's try a different approach: design a path that keeps the sofa (grid) entirely in the corridor.
    # The sofa occupies a 4x1x1 box in local coordinates (x from -4 to 0, y from 0 to 1, z from 0 to 1).
    # We need to move this box from the left arm of the L to the right arm.
    
    # The L-shaped corridor:
    # Left arm: x <= 0, 0 <= y <= 1, 0 <= z <= 1
    # Bottom: 0 <= x <= 1, 0 <= y <= 5, 0 <= z <= 1
    # Right arm: 0 <= x <= 1, 5 <= y <= 6, z >= 0
    
    # Strategy: 
    # 1. Start with sofa in left arm (x from -4 to 0, y from 0 to 1, z from 0 to 1)
    # 2. Move sofa right into the bottom section
    # 3. Rotate and move to the right arm
    
    # The key challenge is the corner. The sofa is 4 units long, so it can't fit in the corner easily.
    # We need to rotate it as it goes around the corner.
    
    # Let's design a path:
    # Phase 1: Move from (-3,0,0) to (1,0,0) - moving right, sofa still horizontal
    #   But the sofa is 4 units long, so when centered at x=1, it extends from x=-1 to x=3.
    #   x=3 is not in the corridor (corridor has x<=1 in the bottom).
    #   So we need to start rotating before we reach x=1.
    
    # Actually, the sofa's local x is from -4 to 0. When at pose (x,y,z,0,0,0), the sofa occupies
    # world x from x-4 to x, y from y to y+1, z from z to z+1.
    
    # For the sofa to be in the left arm: x-4 <= 0 (always true if x <= 4), and we need y in [0,1], z in [0,1].
    # Actually, the sofa occupies x from x-4 to x. For it to be in {x<=0, 0<=y<=1, 0<=z<=1}, we need x <= 0.
    # Wait, no. The sofa occupies a region. For ALL points to be in the corridor, we need the entire sofa to be in the corridor.
    
    # Hmm, but the task says "A grid point survives if it is in the corridor at the first pose and after each move".
    # So we need each grid point to be in the corridor at each pose.
    
    # Let me reconsider. The grid is 81x21x21 = 35721 points. We need all of them to be in the corridor at each pose.
    # That's a very strong constraint.
    
    # Actually wait - "Volume = 0.05^3 times the number of 4x4x4 blocks of adjacent grid points that all survive."
    # So we don't need ALL points to survive, just 4x4x4 blocks. We want to maximize the number of such blocks.
    
    # This is much more flexible. We can sacrifice some points as long as we keep enough 4x4x4 blocks intact.
    
    # Let's try to design a path that keeps as many points as possible in the corridor.
    
    # For now, let's just return the seed path and see if we can improve it with local search.
    
    # Seed path
    seed_path = np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
        [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
        [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
        [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
        [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
        [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
        [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
        [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
        [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
        [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
        [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
        [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
        [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
        [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
        [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
        [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
        [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
        [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
        [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
        [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
        [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
        [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
        [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
        [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
        [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
        [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
        [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
        [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
        [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
        [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
        [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
        [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
        [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
        [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
        [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
        [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
        [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
        [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
        [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
        [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 20.9723911, 5.66341348],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 21.6229248, 5.80874653],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676],
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
        [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
        [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
        [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
        [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
        [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
        [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
        [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
        [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
        [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
        [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
        [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
        [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
        [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
        [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
        [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
        [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
        [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
        [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
        [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
        [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
        [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
        [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979]
    ], dtype=np.float64)
    
    # Try local search to improve the path
    # Perturb individual poses and keep improvements
    
    def count_surviving_blocks(path):
        """Count 4x4x4 blocks of grid points that survive."""
        x_grid = np.linspace(-4, 0, 81)
        y_grid = np.linspace(0, 1, 21)
        z_grid = np.linspace(0, 1, 21)
        
        # Full path with endpoints
        full_path = np.vstack([[-3, 0, 0, 0, 0, 0], path, [0, 6, 7, 90, 90, 90]])
        
        # For each grid point, check if it's in the corridor at all poses
        # Grid point (i,j,k) has local coords (x_grid[i], y_grid[j], z_grid[k])
        
        # For efficiency, check a subset of poses (every 5th)
        poses_to_check = full_path[::5]
        
        # Precompute rotation matrices
        R_list = []
        T_list = []
        for pose in poses_to_check:
            x, y, z, yaw, pitch, roll = pose
            yaw_rad = np.radians(yaw)
            pitch_rad = np.radians(pitch)
            roll_rad = np.radians(roll)
            
            cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
            cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
            cr, sr = np.cos(roll_rad), np.sin(roll_rad)
            
            # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R = Rx @ Ry @ Rz
            
            R_list.append(R)
            T_list.append(np.array([x, y, z]))
        
        # Check each grid point
        # For a grid point to survive, it must be in the corridor at all checked poses
        # And the rotation arc bounding box must be in the corridor (approximate by checking endpoints)
        
        # For speed, let's check all grid points but use vectorized ops
        # Actually, 81*21*21 = 35721 points, and ~20 poses to check
        # That's ~700K checks, should be fast
        
        survive = np.ones((81, 21, 21), dtype=bool)
        
        for R, T in zip(R_list, T_list):
            # Transform all grid points
            # Local: (x_grid[i], y_grid[j], z_grid[k])
            # World: R @ local + T
            
            # Vectorized
            X = x_grid[:, None, None]  # (81, 1, 1)
            Y = y_grid[None, :, None]  # (1, 21, 1)
            Z = z_grid[None, None, :]  # (1, 1, 21)
            
            # R @ [x, y, z]^T + T
            # R is 3x3
            wx = R[0,0]*X + R[0,1]*Y + R[0,2]*Z + T[0]
            wy = R[1,0]*X + R[1,1]*Y + R[1,2]*Z + T[1]
            wz = R[2,0]*X + R[2,1]*Y + R[2,2]*Z + T[2]
            
            # Check corridor
            in_corridor = (
                ((wx <= 0) & (wy >= 0) & (wy <= 1) & (wz >= 0) & (wz <= 1)) |
                ((wx >= 0) & (wx <= 1) & (wy >= 0) & (wy <= 5) & (wz >= 0) & (wz <= 1)) |
                ((wx >= 0) & (wx <= 1) & (wy >= 5) & (wy <= 6) & (wz >= 0))
            )
            
            survive &= in_corridor
        
        # Count 4x4x4 blocks where all points survive
        # A block is defined by (i,j,k) to (i+3,j+3,k+3)
        # We need survive[i:i+4, j:j+4, k:k+4].all()
        
        block_count = 0
        for i in range(81 - 3):
            for j in range(21 - 3):
                for k in range(21 - 3):
                    if survive[i:i+4, j:j+4, k:k+4].all():
                        block_count += 1
        
        return block_count
    
    # This is too slow for local search. Let's use a faster approximation.
    # Just check a few key poses and a subset of grid points.
    
    def fast_evaluate(path):
        """Fast evaluation using a subset of poses and grid points."""
        full_path = np.vstack([[-3, 0, 0, 0, 0, 0], path, [0, 6, 7, 90, 90, 90]])
        
        # Check every 10th pose
        poses_to_check = full_path[::10]
        
        # Check a subset of grid points (every 5th in each dimension)
        x_sub = np.linspace(-4, 0, 17)  # 81/5 ~ 16
        y_sub = np.linspace(0, 1, 5)
        z_sub = np.linspace(0, 1, 5)
        
        X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        
        n_points = len(X)
        n_poses = len(poses_to_check)
        
        # For each pose, check how many points are in the corridor
        min_in_corridor = n_points
        for pose in poses_to_check:
            x, y, z, yaw, pitch, roll = pose
            yaw_rad = np.radians(yaw)
            pitch_rad = np.radians(pitch)
            roll_rad = np.radians(roll)
            
            cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
            cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
            cr, sr = np.cos(roll_rad), np.sin(roll_rad)
            
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R = Rx @ Ry @ Rz
            
            # Transform points
            points = np.vstack([X, Y, Z]).T  # (n_points, 3)
            world_points = points @ R.T + np.array([x, y, z])
            
            wx = world_points[:, 0]
            wy = world_points[:, 1]
            wz = world_points[:, 2]
            
            in_corridor = (
                ((wx <= 0) & (wy >= 0) & (wy <= 1) & (wz >= 0) & (wz <= 1)) |
                ((wx >= 0) & (wx <= 1) & (wy >= 0) & (wy <= 5) & (wz >= 0) & (wz <= 1)) |
                ((wx >= 0) & (wx <= 1) & (wy >= 5) & (wy <= 6) & (wz >= 0))
            )
            
            min_in_corridor = min(min_in_corridor, in_corridor.sum())
        
        return min_in_corridor
    
    # Try local search
    best_path = seed_path.copy()
    best_score = fast_evaluate(best_path)
    
    # Perturb poses
    np.random.seed(42)
    n_iterations = 0
    max_iterations = 5000
    
    while n_iterations < max_iterations and time.time() < deadline - 60:
        # Pick a random pose to perturb
        idx = np.random.randint(len(best_path))
        perturbation = np.random.normal(0, 0.05, 6)
        new_path = best_path.copy()
        new_path[idx] += perturbation
        
        new_score = fast_evaluate(new_path)
        if new_score > best_score:
            best_path = new_path
            best_score = new_score
        
        n_iterations += 1
    
    return best_path
